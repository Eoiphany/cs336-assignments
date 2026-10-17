"""由 Codex 编写的边界回归验证：uv run test/edge_checks.py。

验证非整块、矩形、因果及 FP16 attention；优化器闭包、参数组生成器、
动态学习率和恢复语义。所有输入为小型合成数据，无训练或性能结论。
"""

import torch

from cs336_systems.ddp.shared_optimizer import SharedOptimizer
from cs336_systems.flash_pytorch import FlashAttention2Pytorch


def check_attention():
    # 由 Codex 实现：与原生 autograd 公式比较输出及全部 Q/K/V 梯度。
    torch.manual_seed(10)
    count = 0
    for dtype in (torch.float32, torch.float16):
        for nq, nk in ((35, 70), (70, 35)):
            for causal in (False, True):
                inputs = [torch.randn(2, n, 16, dtype=dtype, requires_grad=True) for n in (nq, nk, nk)]
                q, k, v = inputs
                expected_inputs = [x.detach().float().requires_grad_() for x in inputs]
                qr, kr, vr = expected_inputs
                scores = qr @ kr.transpose(-1, -2) / 4
                if causal:
                    mask = torch.arange(nq)[:, None] < torch.arange(nk)[None, :]
                    scores = scores.masked_fill(mask, -1e6)
                expected = torch.softmax(scores, -1) @ vr
                actual = FlashAttention2Pytorch.apply(q, k, v, causal)
                upstream = torch.randn_like(actual)
                actual.backward(upstream)
                expected.backward(upstream.float())
                torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-3)
                for actual_input, expected_input in zip(inputs, expected_inputs):
                    torch.testing.assert_close(actual_input.grad.float(), expected_input.grad, atol=3e-3, rtol=3e-3)
                count += 1
    print(f"attention boundary cases: {count} passed")


def check_optimizer():
    # 由 Codex 实现：真实修改外层 lr，验证分片本地优化器不会用过时超参数。
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = SharedOptimizer(({"params": [parameter], "lr": 0.5} for _ in range(1)), torch.optim.SGD, lr=1.0)
    optimizer.param_groups[0]["lr"] = 0.2
    calls = []

    def closure():
        # 由 Codex 实现：优化器每步只能重算一次指定的闭包。
        calls.append(1)
        parameter.grad = torch.ones_like(parameter)
        return parameter.sum()

    optimizer.step(closure)
    assert len(calls) == 1
    torch.testing.assert_close(parameter.detach(), torch.tensor([0.8]))
    optimizer.param_groups[0]["lr"] = 0.1
    state = optimizer.state_dict()
    optimizer.param_groups[0]["lr"] = 0.9
    optimizer.load_state_dict(state)
    assert optimizer.param_groups[0]["lr"] == 0.1
    optimizer.step()
    torch.testing.assert_close(parameter.detach(), torch.tensor([0.7]))
    print("optimizer closure/generator/lr/restore: passed")


if __name__ == "__main__":
    check_attention()
    check_optimizer()
