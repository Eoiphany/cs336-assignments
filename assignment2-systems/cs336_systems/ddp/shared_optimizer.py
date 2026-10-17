"""由 Codex 从用户指定参考目录迁移：完成作业要求的实现；保留参考算法。"""

from typing import Any
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.optim import Optimizer


ParamGroup = dict[str, Any]
ParamsLike = Iterable[torch.Tensor] | Iterable[ParamGroup]


def _dist_info() -> tuple[int, int]:
    """Return (rank, world_size). If dist not initialized, treat as single-process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _normalize_param_groups(params: ParamsLike) -> list[ParamGroup]:
    """由 Codex 修复参考实现：参数组也可以是一次性的生成器。"""
    items = list(params)
    if items and isinstance(items[0], dict):
        return [dict(group) for group in items]
    return [{"params": items}]


class SharedOptimizer(Optimizer):
    """
    Optimizer state sharding
      - shard parameters across ranks
      - each rank only maintains optimizer state for its shard
      - after local step, broadcast updated parameters from their owning rank

    Sharding rule: global_param_index % world_size == owning_rank
    Global index is defined by first-seen order across param groups
    """

    def __init__(self, params: ParamsLike, optimizer_cls: type[Optimizer], **kwargs):
        self.rank, self.world_size = _dist_info()
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = dict(kwargs)

        # Full (unshared) param groups
        normalized_groups = _normalize_param_groups(params)

        # Track unique params in a stable global order
        self._global_params: list[torch.Tensor] = []
        self._param_to_gidx: dict[torch.Tensor, int] = {}

        # Local (shared) param groups used to construct the *real* optimizer on this rank
        self._local_param_groups: list[ParamGroup] = []

        # IMPORTANT: call parent ctor
        super().__init__(normalized_groups, defaults=dict(kwargs))

        # Construct the local optimizer that only sees local params
        self._local_optimizer: Optimizer = self._optimizer_cls(self._local_param_groups, **self._optimizer_kwargs)

    def add_param_group(self, param_group: ParamGroup) -> None:
        """
        1) register the full param group into self.param_groups
        2) create a filtered local param group (only params owned by this rank) for local optimizer
        """
        # --- normalize and validate group
        if "params" not in param_group:
            raise ValueError("param_group must have a 'params' key")
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            raise TypeError("param_group['params'] must be an iterable of Parameters, not a Tensor")
        params_list = list(params)

        # Register full group into this Optimizer
        full_group = dict(param_group)
        full_group["params"] = params_list
        # Use base class machinery to keep invariants
        super().add_param_group(full_group)

        # --- update global param ordering / indexing
        for p in params_list:
            if p not in self._param_to_gidx:
                self._param_to_gidx[p] = len(self._global_params)
                self._global_params.append(p)

        # --- build local (sharded) group with same hyperparams, but only owned params
        local_params: list[torch.Tensor] = []
        seen_local: set[torch.Tensor] = set()
        for p in params_list:
            gidx = self._param_to_gidx[p]
            owner = gidx % self.world_size
            if owner == self.rank and p not in seen_local:
                local_params.append(p)
                seen_local.add(p)

        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params
        self._local_param_groups.append(local_group)

        # If local optimizer already exists, update it too
        if hasattr(self, "_local_optimizer"):
            # torch optimizers support add_param_group
            self._local_optimizer.add_param_group(local_group)

    @torch.no_grad()
    def _broadcast_updated_parameters(self) -> None:
        """Broadcast each parameter tensor from its owning rank to all ranks."""
        if self.world_size == 1:
            return

        # Deterministic order matters for debug/repor, but broadcast itself is per-tensor collective.
        for p in self._global_params:
            owner = self._param_to_gidx[p] % self.world_size
            dist.broadcast(p.data, src=owner)

    def step(self, closure=None, **kwargs):
        """
        1) run closure (if provided) to compute loss
        2) local optimizer step (updates only owned params)
        3) broadcast update params from owners
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Call local optimizer step
        # 由 Codex 修复：closure 已执行一次，不在本地优化器中再次调用。
        self._sync_param_group_options()
        self._local_optimizer.step(**kwargs)

        # Sync parameters across ranks
        self._broadcast_updated_parameters()

        if self.world_size > 1:
            dist.barrier()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Important: must clear grads for *all* parameters (not only local shard),
        otherwise behavior diverges from baseline optimizer in tests.
        """
        seen: set[torch.Tensor] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p in seen:
                    continue
                seen.add(p)
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    def state_dict(self) -> dict[str, Any]:
        """由 Codex 修复：先同步 scheduler 的最新超参数，再保存本地分片。"""
        self._sync_param_group_options()
        return {"rank": self.rank, "world_size": self.world_size, "local_optimizer": self._local_optimizer.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """由 Codex 修复：恢复本地分片状态，并同步公开参数组的超参数。"""
        if "local_optimizer" not in state_dict:
            raise ValueError("Missing local_optimizer state")
        self._local_optimizer.load_state_dict(state_dict["local_optimizer"])
        for outer, local in zip(self.param_groups, self._local_optimizer.param_groups):
            outer.update({key: value for key, value in local.items() if key != "params"})

    def _sync_param_group_options(self):
        """由 Codex 实现：公开参数组是 scheduler 和用户修改的唯一超参数来源。"""
        for outer, local in zip(self.param_groups, self._local_optimizer.param_groups):
            local.update({key: value for key, value in outer.items() if key != "params"})
