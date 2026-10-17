"""Codex适配参考实现：本地JSONL指标记录；仅显式启用时连接W&B。"""

from dataclasses import asdict, is_dataclass
import json
import os
import tempfile
from pathlib import Path
import time


class ExperimentTracker:
    """Codex适配：记录global step和累计秒数，恢复时在检查点之前的历史后追加。"""

    def __init__(
        self,
        run_dir,
        config,
        *,
        use_wandb=False,
        wandb_project="cs336-a1",
        wandb_run_name=None,
        elapsed_seconds=0.0,
        resume=False,
        resume_step=0,
    ):
        """Codex适配：新run写配置，resume保留原始配置；W&B默认关闭。"""
        self.run_dir = Path(run_dir)
        self.t0 = time.monotonic()
        self.elapsed_seconds = elapsed_seconds
        self.metrics_path = self.run_dir / "metrics.jsonl"
        if not resume:
            with (self.run_dir / "config.json").open("x", encoding="utf-8") as handle:
                json.dump(asdict(config) if is_dataclass(config) else config, handle, indent=2)
        if resume and self.metrics_path.exists():
            # Discard metrics from an interrupted epoch that is about to be replayed.
            records = []
            for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A process interruption may leave a partial final line.
                if record["step"] <= resume_step:
                    records.append(record)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.run_dir, delete=False) as handle:
                for record in records:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                temporary = Path(handle.name)
            os.replace(temporary, self.metrics_path)
        self._wandb = None
        if use_wandb:
            import wandb

            self._wandb = wandb.init(
                project=wandb_project, name=wandb_run_name, config=asdict(config), dir=str(self.run_dir)
            )

    def wall_time_s(self):
        """Codex适配：返回本次与历史训练累计耗时，单位秒。"""
        return self.elapsed_seconds + time.monotonic() - self.t0

    def log(self, step, metrics):
        """Codex适配：追加一个标量字典；不保存模型或隐式触发训练。"""
        record = {"step": int(step), "wall_time_s": self.wall_time_s(), **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if self._wandb is not None:
            self._wandb.log(record, step=step)

    def close(self):
        """Codex适配：仅关闭本对象启动的W&B run，本地指标已逐行落盘。"""
        if self._wandb is not None:
            self._wandb.finish()
