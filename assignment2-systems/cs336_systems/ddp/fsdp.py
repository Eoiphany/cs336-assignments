"""由 Codex 实现手册的 FSDP：线性/嵌入权重分片，其他参数复制。

在项目目录运行 ``uv run pytest -q tests/test_fsdp.py`` 验证。各 rank 保留
FP32 主权重分片；按层聚合计算权重，反向时重聚合，异步 reduce-scatter 梯度。
本实现采用即时聚合，没有预取未来层，因此不会提前保留额外层的完整权重。
"""

from dataclasses import dataclass, field
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class _GradientQueue:
    """由 Codex 实现：只保留最近一次异步归约，限制完整梯度通信缓冲的峰值。"""

    def __init__(self):
        self.inflight = None

    def drain(self):
        # 由 Codex 修复：完成上一层通信后及时释放其完整缓冲，而非等到整网反向结束。
        if self.inflight is not None:
            self.inflight.finish()
            self.inflight = None


@dataclass
class _WeightShard:
    parameter: nn.Parameter
    shape: torch.Size
    world_size: int
    compute_dtype: torch.dtype | None
    communication: _GradientQueue
    pending: list = field(default_factory=list)

    def gather(self, *, master=False):
        """由 Codex 实现：聚合等长分片并裁去末尾填充，返回原矩阵形状。"""
        local = self.parameter.detach().to(self.parameter.dtype if master else self.compute_dtype or self.parameter.dtype)
        if self.world_size == 1:
            flat = local
        else:
            parts = [torch.empty_like(local) for _ in range(self.world_size)]
            dist.all_gather(parts, local)
            flat = torch.cat(parts)
        return flat[: self.shape.numel()].view(self.shape)

    def queue_gradient(self, full_gradient):
        """由 Codex 实现：保留通信缓冲区至同步点，异步归约到本地 FP32 分片。"""
        self.communication.drain()
        size = self.parameter.numel()
        padded = full_gradient.new_zeros(size * self.world_size, dtype=self.parameter.dtype)
        padded[: self.shape.numel()].copy_(full_gradient.reshape(-1))
        output = torch.empty_like(self.parameter)
        if self.world_size == 1:
            output.copy_(padded)
            work = None
        else:
            work = dist.reduce_scatter_tensor(output, padded, async_op=True)
        self.pending.append((work, output, padded))
        self.communication.inflight = self

    def finish(self):
        """由 Codex 实现：等待通信并累加平均梯度，兼容共享权重多次使用。"""
        for work, output, _buffer in self.pending:
            if work is not None:
                work.wait()
            output.div_(self.world_size)
            if self.parameter.grad is None:
                self.parameter.grad = output
            else:
                self.parameter.grad.add_(output)
        self.pending.clear()


class _ShardedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, local_weight, bias, shard):
        # 由 Codex 实现：只保存输入和本地分片，不把完整权重留给 autograd。
        weight = shard.gather()
        ctx.save_for_backward(inputs, local_weight)
        ctx.shard = shard
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return F.linear(inputs.to(weight.dtype), weight, bias.to(weight.dtype) if bias is not None else None)

    @staticmethod
    def backward(ctx, grad_output):
        # 由 Codex 实现：重新聚合权重计算输入梯度，完整权重梯度仅用于归约。
        inputs, _local_weight = ctx.saved_tensors
        weight = ctx.shard.gather()
        grad = grad_output.to(weight.dtype)
        dx = (grad @ weight).to(inputs.dtype) if ctx.needs_input_grad[0] else None
        if ctx.needs_input_grad[1]:
            dw = grad.reshape(-1, weight.shape[0]).T @ inputs.to(weight.dtype).reshape(-1, weight.shape[1])
            ctx.shard.queue_gradient(dw)
        db = grad.reshape(-1, weight.shape[0]).sum(0).to(ctx.bias_dtype) if ctx.needs_input_grad[2] else None
        return dx, None, db, None


class _ShardedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, local_weight, shard, padding_idx, scale_grad_by_freq):
        # 由 Codex 实现：输出 (..., embedding_dim)，完整矩阵在本次前向结束后释放。
        weight = shard.gather()
        ctx.save_for_backward(indices)
        ctx.shard = shard
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        return F.embedding(indices, weight, padding_idx=padding_idx)

    @staticmethod
    def backward(ctx, grad_output):
        # 由 Codex 实现：按 token 索引累加行梯度，保留 padding 与频次缩放语义。
        (indices,) = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            ids = indices.reshape(-1)
            gradient = grad_output.reshape(-1, ctx.shard.shape[1])
            if ctx.scale_grad_by_freq:
                counts = torch.bincount(ids, minlength=ctx.shard.shape[0])
                gradient = gradient / counts[ids, None]
            dw = gradient.new_zeros(ctx.shard.shape)
            dw.index_add_(0, ids, gradient)
            if ctx.padding_idx is not None:
                dw[ctx.padding_idx].zero_()
            ctx.shard.queue_gradient(dw)
        return None, None, None, None, None


def _linear_forward(module, inputs):
    # 由 Codex 实现：保留原层实例及参数名，只替换计算路径。
    return _ShardedLinear.apply(inputs, module.weight, getattr(module, "bias", None), module._fsdp_shard)


def _embedding_forward(module, indices):
    # 由 Codex 实现：官方层与 torch.nn.Embedding 共用分片实现。
    return _ShardedEmbedding.apply(indices, module.weight, module._fsdp_shard, getattr(module, "padding_idx", None), getattr(module, "scale_grad_by_freq", False))


class FullyShardedDataParallel(nn.Module):
    def __init__(self, module, compute_dtype=None):
        """由 Codex 实现：广播初始状态，分片 Linear/Embedding，其余参数保持复制。"""
        super().__init__()
        from cs336_basics.model import Embedding, Linear

        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before constructing FSDP.")
        self.module = module
        self.world_size = dist.get_world_size()
        self._shards = {}
        self._communication = _GradientQueue()
        rank = dist.get_rank()
        for parameter in module.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.float()
            dist.broadcast(parameter.data, src=0)
        for buffer in module.buffers():
            dist.broadcast(buffer, src=0)
        for layer in module.modules():
            if not isinstance(layer, (Linear, Embedding, nn.Linear, nn.Embedding)):
                continue
            is_embedding = isinstance(layer, (Embedding, nn.Embedding))
            if is_embedding and (getattr(layer, "max_norm", None) is not None or getattr(layer, "sparse", False)):
                raise ValueError("FSDP embedding supports dense gradients and max_norm=None.")
            parameter = layer.weight
            if id(parameter) not in self._shards:
                shape = parameter.shape
                shard_size = (parameter.numel() + self.world_size - 1) // self.world_size
                padded = parameter.detach().new_zeros(shard_size * self.world_size)
                padded[: parameter.numel()].copy_(parameter.detach().reshape(-1))
                parameter.data = padded[rank * shard_size : (rank + 1) * shard_size].clone()
                self._shards[id(parameter)] = _WeightShard(parameter, shape, self.world_size, compute_dtype, self._communication)
            layer._fsdp_shard = self._shards[id(parameter)]
            layer.forward = MethodType(_embedding_forward if is_embedding else _linear_forward, layer)

    def forward(self, *inputs, **kwargs):
        # 由 Codex 实现：保持原模型的调用参数与输出不变。
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        """由 Codex 实现：完成分片梯度归约，同时平均未分片参数的梯度。"""
        self._communication.drain()
        for shard in self._shards.values():
            shard.finish()
        replicated = []
        for parameter in self.module.parameters():
            if id(parameter) not in self._shards and parameter.grad is not None:
                work = dist.all_reduce(parameter.grad, async_op=True)
                replicated.append((work, parameter.grad))
        for work, gradient in replicated:
            work.wait()
            gradient.div_(self.world_size)

    @torch.no_grad()
    def gather_full_params(self):
        """由 Codex 实现：按原模型参数名导出完整 FP32 权重用于核验。"""
        return {name: self._shards[id(p)].gather(master=True).clone() if id(p) in self._shards else p.detach().clone() for name, p in self.module.named_parameters()}
