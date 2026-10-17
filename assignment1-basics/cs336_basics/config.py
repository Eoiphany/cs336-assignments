"""Codex 适配参考实现：训练JSON配置、模型构造及可复现参数目录。

训练入口：uv run python cs336_basics/train.py --help（macOS/Windows CMD相同）。
配置分组为run/data/model/optim/train/wandb；字段默认值见下列dataclass。
数据路径默认空，必须明确提供；默认CPU、1个完整epoch、关闭W&B。
"""

from dataclasses import dataclass, field, fields
import json
from pathlib import Path

import torch


@dataclass
class RunConfig:
    """Codex适配：输出为runs_dir/experiment_name/<下划线参数>。"""

    runs_dir: str = "runs"
    experiment_name: str = "language_model"
    parameter_name: str | None = None


@dataclass
class DataConfig:
    """Codex适配：已划分的1D原始整数token文件，路径必须显式配置。"""

    train_data_path: str = ""
    val_data_path: str = ""
    np_dtype: str = "uint16"
    context_length: int = 256
    device: str = "cpu"


@dataclass
class ModelConfig:
    """Codex适配：模型维度为token个数/隐层通道数，dtype为参数精度。"""

    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int | None = 1344
    rope_theta: float = 10_000.0
    max_seq_len: int | None = None
    rmsnorm_eps: float = 1e-5
    torch_dtype: str = "float32"


@dataclass
class OptimizerConfig:
    """Codex适配：学习率/衰减为无量纲，warmup/cosine周期单位为更新步。"""

    lr_max: float = 1e-3
    lr_min: float = 1e-4
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 106_667
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0


@dataclass
class TrainingConfig:
    """Codex适配：epochs为总目标epoch；恢复时可增加。检查点仅按完整epoch保存。"""

    epochs: int = 1
    batch_size: int = 12
    log_interval: int = 50
    eval_batches: int = 20
    seed: int = 42


@dataclass
class WandbConfig:
    """Codex适配：只有显式enable=true才初始化W&B；默认完全本地。"""

    enable: bool = False
    project: str = "cs336-a1"
    run_name: str | None = None


@dataclass
class TrainConfig:
    """Codex适配：序列化完整训练配置，避免依赖脚本硬编码超参数。"""

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def config_from_dict(value: dict) -> TrainConfig:
    """Codex实现：读取已知配置分组；未知键直接报错，防止静默忽略参数。"""
    types = {
        "run": RunConfig,
        "data": DataConfig,
        "model": ModelConfig,
        "optim": OptimizerConfig,
        "train": TrainingConfig,
        "wandb": WandbConfig,
    }
    if not isinstance(value, dict) or set(value) - set(types):
        raise ValueError("Config must contain only run/data/model/optim/train/wandb objects")
    groups = {}
    for name, cls in types.items():
        group = value.get(name, {})
        if not isinstance(group, dict) or set(group) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown or invalid keys in config group {name}")
        groups[name] = cls(**group)
    return TrainConfig(**groups)


def load_config(path: str | Path) -> TrainConfig:
    """Codex实现：从UTF-8 JSON加载配置；相对数据路径相对于当前目录。"""
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def get_default_config() -> TrainConfig:
    """Codex适配：返回独立默认值；不开始训练、不创建输出或联网。"""
    return TrainConfig()


def torch_dtype_from_string(name: str) -> torch.dtype:
    """Codex适配：将显式精度名称映射至PyTorch dtype，未知名称报错。"""
    names = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name.lower() not in names:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return names[name.lower()]


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    """Codex适配：模型输入[B,T] token IDs，输出[B,T,vocab_size] logits。"""
    from cs336_basics.transformer_lm import TransformerLM

    model = cfg.model
    return TransformerLM(
        vocab_size=model.vocab_size,
        context_length=model.context_length,
        d_model=model.d_model,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        d_ff=model.d_ff if model.d_ff is not None else 4 * model.d_model,
        rope_theta=model.rope_theta,
        max_seq_len=model.max_seq_len,
        eps=model.rmsnorm_eps,
        device=torch.device(cfg.data.device),
        dtype=torch_dtype_from_string(model.torch_dtype),
    )


def parameter_directory(cfg: TrainConfig) -> Path:
    """Codex实现：生成runs/<实验>/<下划线连接参数>路径，不自动覆盖已有run。"""
    name = cfg.run.parameter_name or (
        f"{cfg.train.batch_size}_{cfg.optim.lr_max}_{cfg.model.num_layers}_"
        f"{cfg.model.d_model}_{cfg.model.context_length}_{cfg.train.seed}"
    )
    for part in (cfg.run.experiment_name, name):
        if not part or Path(part).name != part or part in (".", ".."):
            raise ValueError("Experiment and parameter names must be single path components")
    return Path(cfg.run.runs_dir) / cfg.run.experiment_name / name
