"""由 Codex 适配参考计时流程，使用本版官方 BasicsTransformerLM。

项目目录示例：uv run cs336_systems/benchmark.py --model-size tiny --mode train
多进程：uv run torchrun --standalone --nproc-per-node=2 cs336_systems/benchmark.py --parallel fsdp
--mode 为 forward/backward/train；--warmup/--steps 控制预热/测量次数；
--dtype 控制计算精度（主权重保持 FP32），--compile 启用 torch.compile；
模型尺寸、batch、序列长度均可配置。输出本次真实耗时 JSON，不执行外部提交。
"""

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


SPECS = {
    "tiny": (32, 64, 2, 4),  # 本地功能验证；不作为手册标准模型的性能结果。
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (1600, 6400, 48, 25),
    "2.7b": (2560, 10240, 32, 32),
}


def main():
    """由 Codex 实现：配置模型并计时，可在 torchrun 下测量 DDP/FSDP。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=SPECS, default="tiny")
    parser.add_argument("--batch-size", type=int, default=2, help="每个 rank 的 batch size")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--mode", choices=["forward", "backward", "train"], default="train", help="backward 包含前向与反向")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--parallel", choices=["none", "ddp", "fsdp"], default="none")
    parser.add_argument("--shard-optimizer", action="store_true")
    parser.add_argument("--compile", action="store_true", help="启用编译；预热时间不计入测量")
    args = parser.parse_args()
    if min(args.batch_size, args.context_length, args.steps) <= 0 or args.warmup < 0:
        parser.error("batch-size/context-length/steps must be positive; warmup must be nonnegative")
    if args.shard_optimizer and args.parallel != "ddp":
        parser.error("--shard-optimizer requires --parallel ddp")
    device = torch.device(args.device)
    distributed = args.parallel != "none"
    if distributed:
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        rank = dist.get_rank() if distributed else 0
        torch.manual_seed(0)
        d_model, d_ff, num_layers, num_heads = SPECS[args.model_size]
        model = BasicsTransformerLM(args.vocab_size, args.context_length, d_model, num_layers, num_heads, d_ff).to(device)
        dtype = getattr(torch, args.dtype)
        if args.compile:
            model = torch.compile(model)
        if args.parallel == "ddp":
            from cs336_systems.ddp.ddp_overlap_individual_parameters import DDPIndividualParameters

            model = DDPIndividualParameters(model)
        elif args.parallel == "fsdp":
            from cs336_systems.ddp.fsdp import FullyShardedDataParallel

            model = FullyShardedDataParallel(model, None if dtype == torch.float32 else dtype)
        if args.shard_optimizer:
            from cs336_systems.ddp.shared_optimizer import SharedOptimizer

            optimizer = SharedOptimizer(model.parameters(), AdamW, lr=1e-3)
        else:
            optimizer = AdamW(model.parameters(), lr=1e-3)
        torch.manual_seed(rank + 1)
        inputs = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
        labels = torch.randint(args.vocab_size, inputs.shape, device=device)

        def sync():
            # 由 Codex 实现：主机计时前后等待设备，避免只测到异步提交开销。
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
            if distributed:
                dist.barrier()

        times = []
        for step in range(args.warmup + args.steps):
            optimizer.zero_grad(set_to_none=True)
            sync()
            started = time.perf_counter()
            amp = torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 and args.parallel != "fsdp" else nullcontext()
            grad = torch.no_grad() if args.mode == "forward" else nullcontext()
            with grad, amp:
                output = model(inputs)
                if args.mode != "forward":
                    loss = cross_entropy(output.float(), labels).mean()
            if args.mode != "forward":
                loss.backward()
                if distributed:
                    model.finish_gradient_synchronization()
                if args.mode == "train":
                    optimizer.step()
            sync()
            if step >= args.warmup:
                times.append((time.perf_counter() - started) * 1000)
        if rank == 0:
            print(json.dumps({"config": vars(args), "mean_ms": statistics.mean(times), "std_ms": statistics.pstdev(times), "samples_ms": times}, indent=2))
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
