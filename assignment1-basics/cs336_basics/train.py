"""Codex适配参考训练：memmap完整epoch、原子检查点与可复现恢复。

macOS/Windows CMD均可：
  uv run python cs336_basics/train.py --config train.json
  uv run python cs336_basics/train.py --resume runs/实验/参数/last.pt --epochs 2
--config为config.py的JSON分组；--train-data/--val-data/--device/--epochs/
--runs-dir/--experiment-name可覆盖对应字段。默认CPU、1 epoch、关闭W&B；
数据路径必须明确提供。每个epoch遍历一次不重叠context窗口并随机打乱，
不足一个context的末尾token不训练。验证集单独memmap，不重新划分。
输出：runs/<实验>/<下划线参数>/checkpoints/epoch_XXXX.pt完整状态，
同级last.pt完整状态、best.pt纯state_dict、history.json和metrics.jsonl。
恢复仅在完整epoch边界进行；中断的epoch从最近完整检查点重新训练。
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cs336_basics.config import (
    build_model,
    config_from_dict,
    get_default_config,
    load_config,
    parameter_directory,
    torch_dtype_from_string,
)
from cs336_basics.data import get_batch
from cs336_basics.experiment_tracking import ExperimentTracker
from cs336_basics.nn_utils import clip_grad_norm, cross_entropy_from_logits
from cs336_basics.optimizer import AdamW, lr_cosine_schedule_with_warmup


def open_memmap_1d(path, np_dtype):
    """Codex适配：只读映射原始整数token；文件大小必须是元素字节数的整数倍。"""
    if not path:
        raise ValueError("Provide explicit train_data_path and val_data_path")
    dtype = np.dtype(np_dtype)
    size = Path(path).stat().st_size
    if dtype.kind not in "iu" or size == 0 or size % dtype.itemsize:
        raise ValueError(f"Invalid integer token file/dtype: {path}, {np_dtype}")
    return np.memmap(path, dtype=dtype, mode="r")


def set_optimizer_lr(optimizer, lr):
    """Codex适配：将当前更新步学习率写入每个参数组，无独立调度器隐式状态。"""
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def estimate_loss(model, data, cfg):
    """Codex适配：用eval_batches个随机验证batch估计nats/token并恢复训练模式。"""
    was_training = model.training
    model.eval()
    try:
        losses = []
        for _ in range(cfg.train.eval_batches):
            x, y = get_batch(data, cfg.train.batch_size, cfg.data.context_length, cfg.data.device)
            if x.min() < 0 or x.max() >= cfg.model.vocab_size or y.max() >= cfg.model.vocab_size:
                raise ValueError("Validation token ID lies outside the configured vocabulary")
            losses.append(float(cross_entropy_from_logits(model(x), y)))
        return float(np.mean(losses))
    finally:
        model.train(was_training)


def random_state():
    """Codex实现：保存Python、NumPy、Torch CPU及实际可用加速器随机状态。"""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_random_state(state):
    """Codex实现：恢复完整epoch结束时随机流；验证和下一epoch顺序均可重现。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        if not torch.backends.mps.is_available():
            raise ValueError("Checkpoint contains MPS RNG state but MPS is unavailable")
        torch.mps.set_rng_state(state["mps"])


def atomic_torch_save(value, path, *, exclusive=False):
    """Codex实现：同目录临时文件+原子替换；epoch归档使用硬链接拒绝覆盖。"""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)  # Atomic, fails if an archive already exists.
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(value, path):
    """Codex实现：原子更新history JSON，避免写入中断留下半个文件。"""
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".history-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def data_descriptor(cfg, train_data, val_data):
    """Codex实现：保存预划分文件身份与窗口规则；恢复时禁止静默换数据/划分。"""
    result = {
        "dtype": cfg.data.np_dtype,
        "context_length": cfg.data.context_length,
        "window_policy": "nonoverlapping_context_windows; discard_incomplete_tail",
    }
    for split, path, data in (
        ("train", cfg.data.train_data_path, train_data),
        ("val", cfg.data.val_data_path, val_data),
    ):
        p = Path(path).resolve()
        stat = p.stat()
        result[split] = {
            "path": str(p),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "tokens": len(data),
            "windows": (len(data) - 1) // cfg.data.context_length,
        }
    return result


def validate_config(cfg):
    """Codex实现：训练开始前检查尺寸、优化器和epoch配置，避免产生无效run。"""
    if cfg.model.context_length != cfg.data.context_length:
        raise ValueError("model.context_length must equal data.context_length")
    if (
        min(
            cfg.train.epochs,
            cfg.train.batch_size,
            cfg.train.eval_batches,
            cfg.train.log_interval,
            cfg.data.context_length,
            cfg.model.vocab_size,
            cfg.model.d_model,
            cfg.model.num_layers,
            cfg.model.num_heads,
        )
        <= 0
    ):
        raise ValueError("Epoch/batch/log/eval/model dimensions must be positive")
    if cfg.model.d_model % cfg.model.num_heads or (cfg.model.d_model // cfg.model.num_heads) % 2:
        raise ValueError("d_model must divide into even-sized attention heads")
    if cfg.model.d_ff is not None and cfg.model.d_ff <= 0:
        raise ValueError("d_ff must be positive or null")
    if cfg.model.max_seq_len is not None and cfg.model.max_seq_len < cfg.model.context_length:
        raise ValueError("max_seq_len must cover context_length")
    if not 0 <= cfg.optim.lr_min <= cfg.optim.lr_max or cfg.optim.lr_max <= 0:
        raise ValueError("Require 0 <= lr_min <= lr_max and positive lr_max")
    if cfg.optim.warmup_iters < 0 or cfg.optim.cosine_cycle_iters <= cfg.optim.warmup_iters:
        raise ValueError("cosine_cycle_iters must exceed nonnegative warmup_iters")
    if cfg.optim.grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    torch_dtype_from_string(cfg.model.torch_dtype)


def train(cfg, resume_from=None):
    """Codex实现：训练完整epoch；返回run目录，绝不按step保存检查点。"""
    validate_config(cfg)
    if cfg.data.train_data_path:
        cfg.data.train_data_path = str(Path(cfg.data.train_data_path).resolve())
    if cfg.data.val_data_path:
        cfg.data.val_data_path = str(Path(cfg.data.val_data_path).resolve())
    train_data = open_memmap_1d(cfg.data.train_data_path, cfg.data.np_dtype)
    val_data = open_memmap_1d(cfg.data.val_data_path, cfg.data.np_dtype)
    descriptor = data_descriptor(cfg, train_data, val_data)
    if min(descriptor["train"]["windows"], descriptor["val"]["windows"]) < 1:
        raise ValueError("Both files need at least context_length+1 tokens")
    checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False) if resume_from else None
    if checkpoint is not None:
        if checkpoint.get("format") != "cs336_epoch_v1":
            raise ValueError("Resume requires this entrypoint's complete epoch checkpoint")
        old = checkpoint["config"]
        current = asdict(cfg)
        for section in ("model", "optim", "train", "wandb", "run"):
            a, b = dict(old[section]), dict(current[section])
            if section == "train":
                a.pop("epochs")
                b.pop("epochs")
            if a != b:
                raise ValueError(f"Resume cannot change {section} (except total epochs)")
        if checkpoint["data_state"] != descriptor:
            raise ValueError("Resume data files/window split differ from saved data_state")
        run_dir = Path(checkpoint["run_dir"])
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Original run directory is missing: {run_dir}")
        start_epoch = checkpoint["epoch"]
        if cfg.train.epochs <= start_epoch:
            raise ValueError("Target epochs must exceed the completed checkpoint epoch")
        if any(run_dir.joinpath("checkpoints").glob("epoch_*.pt")):
            archived = [int(p.stem.split("_")[-1]) for p in (run_dir / "checkpoints").glob("epoch_*.pt")]
            if max(archived) > start_epoch:
                raise ValueError("Newer epoch archives exist; resume from the latest complete checkpoint")
    else:
        run_dir = parameter_directory(cfg).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir()
        start_epoch = 0
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr_max,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    history, step, best, best_epoch, elapsed = [], 0, math.inf, 0, 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history, step = checkpoint["history"], checkpoint["global_step"]
        best, best_epoch = checkpoint["best_val_loss"], checkpoint["best_epoch"]
        elapsed = checkpoint["elapsed_seconds"]
    tracker = ExperimentTracker(
        run_dir,
        cfg,
        use_wandb=cfg.wandb.enable,
        wandb_project=cfg.wandb.project,
        wandb_run_name=cfg.wandb.run_name,
        elapsed_seconds=elapsed,
        resume=checkpoint is not None,
        resume_step=step,
    )
    if checkpoint is not None:
        restore_random_state(checkpoint["random_state"])
    try:
        windows = descriptor["train"]["windows"]
        for epoch in range(start_epoch + 1, cfg.train.epochs + 1):
            archive = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
            if archive.exists():
                raise FileExistsError(f"Refusing to overwrite {archive}")
            model.train()
            weighted_loss, count = 0.0, 0
            order = np.random.permutation(windows)
            for offset in range(0, windows, cfg.train.batch_size):
                starts = order[offset : offset + cfg.train.batch_size] * cfg.data.context_length
                indices = starts[:, None] + np.arange(cfg.data.context_length + 1)[None, :]
                block = np.asarray(train_data[indices], dtype=np.int64)
                if block.min() < 0 or block.max() >= cfg.model.vocab_size:
                    raise ValueError("Training token ID lies outside the configured vocabulary")
                x = torch.as_tensor(block[:, :-1].copy(), device=cfg.data.device)
                y = torch.as_tensor(block[:, 1:].copy(), device=cfg.data.device)
                lr = lr_cosine_schedule_with_warmup(
                    step, cfg.optim.lr_max, cfg.optim.lr_min, cfg.optim.warmup_iters, cfg.optim.cosine_cycle_iters
                )
                set_optimizer_lr(optimizer, lr)
                optimizer.zero_grad(set_to_none=True)
                loss = cross_entropy_from_logits(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at global step {step}")
                loss.backward()
                clip_grad_norm(model.parameters(), cfg.optim.grad_clip)
                optimizer.step()
                step += 1
                weighted_loss += float(loss.detach()) * y.numel()
                count += y.numel()
                if step % cfg.train.log_interval == 0:
                    tracker.log(step, {"epoch": epoch, "train_loss": float(loss.detach()), "lr": lr})
            val_loss = estimate_loss(model, val_data, cfg)
            if not math.isfinite(val_loss):
                raise FloatingPointError("Nonfinite validation loss; epoch was not saved")
            if val_loss < best:
                best, best_epoch = val_loss, epoch
                atomic_torch_save(model.state_dict(), run_dir / "best.pt")
            record = {
                "epoch": epoch,
                "global_step": step,
                "train_loss": weighted_loss / count,
                "val_loss": val_loss,
                "best_val_loss": best,
            }
            history.append(record)
            tracker.log(step, record)
            state = {
                "format": "cs336_epoch_v1",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": step,
                "iteration": step,
                "config": asdict(cfg),
                "data_state": descriptor,
                "history": history,
                "best_val_loss": best,
                "best_epoch": best_epoch,
                "random_state": random_state(),
                "scheduler_state": {"global_step": step, "config": asdict(cfg.optim)},
                "run_dir": str(run_dir),
                "elapsed_seconds": tracker.wall_time_s(),
                "early_stopping_state": None,
                "amp_state": None,
                "ema_state": None,
            }
            atomic_torch_save(state, archive, exclusive=True)
            atomic_torch_save(state, run_dir / "last.pt")
            atomic_json_save(history, run_dir / "history.json")
            print(json.dumps(record, allow_nan=False), flush=True)
    finally:
        tracker.close()
    return run_dir


def main():
    """Codex实现：JSON+显式CLI覆盖；--resume可直接读取检查点内完整配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-data")
    parser.add_argument("--val-data")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--runs-dir")
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
    elif args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = config_from_dict(saved["config"])
    else:
        cfg = get_default_config()
    for group, field, value in (
        (cfg.data, "train_data_path", args.train_data),
        (cfg.data, "val_data_path", args.val_data),
        (cfg.data, "device", args.device),
        (cfg.train, "epochs", args.epochs),
        (cfg.run, "runs_dir", args.runs_dir),
        (cfg.run, "experiment_name", args.experiment_name),
    ):
        if value is not None:
            setattr(group, field, value)
    print(f"Run directory: {train(cfg, args.resume)}")


if __name__ == "__main__":
    main()
