"""由 Codex 编写的双 rank FSDP 回归：uv run test/fsdp_edge_checks.py。

验证标准 PyTorch 层、Embedding/Linear 共享权重、不整除分片、复制 bias、
完整梯度通信缓冲上限及两次更新与非并行模型一致。使用小型合成 CPU 数据。
"""

from copy import deepcopy
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_systems.ddp.fsdp import FullyShardedDataParallel


class TiedModel(nn.Module):
    def __init__(self):
        # 由 Codex 实现：11×7=77 个权重不能被两个 rank 整除，且两层共享同一参数。
        super().__init__()
        self.embedding = nn.Embedding(11, 7, padding_idx=0)
        self.norm = nn.LayerNorm(7)
        self.middle = nn.Linear(7, 7)
        self.output = nn.Linear(7, 11, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, indices):
        # 由 Codex 实现：共享权重的两条梯度路径在本地分片上必须正确累加。
        return self.output(self.middle(self.norm(self.embedding(indices))))


def check_rank(rank, rendezvous):
    # 由 Codex 实现：每个 rank 看一半 batch，与全 batch 基线对照真实更新。
    dist.init_process_group("gloo", init_method=rendezvous, world_size=2, rank=rank)
    try:
        torch.manual_seed(20)
        baseline = TiedModel()
        model = FullyShardedDataParallel(deepcopy(baseline))
        assert model.module.embedding.weight is model.module.output.weight
        assert model.module.embedding.weight.numel() == 39
        expected_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01)
        actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        torch.manual_seed(21)
        indices = torch.randint(0, 11, (4, 3))
        labels = torch.randint(0, 11, (4, 3))
        for _ in range(2):
            expected_optimizer.zero_grad(set_to_none=True)
            actual_optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(baseline(indices).reshape(-1, 11), labels.reshape(-1)).backward()
            local = slice(rank * 2, (rank + 1) * 2)
            nn.functional.cross_entropy(model(indices[local]).reshape(-1, 11), labels[local].reshape(-1)).backward()
            assert sum(len(state.pending) for state in model._shards.values()) <= 1
            model.finish_gradient_synchronization()
            assert not any(state.pending for state in model._shards.values())
            for parameter in model.parameters():
                assert parameter.grad is not None and parameter.grad.shape == parameter.shape
            expected_optimizer.step()
            actual_optimizer.step()
            actual = model.gather_full_params()
            for name, parameter in baseline.named_parameters():
                torch.testing.assert_close(actual[name], parameter, atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(check_rank, args=(Path(directory, "group").as_uri(),), nprocs=2, join=True)
    print("FSDP tied weights/padding/bias/buffer bound: passed on two ranks")
