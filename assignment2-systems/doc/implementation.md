# Assignment 2 实现与验证

日期：2026-10-16（Asia/Shanghai）。由 Codex 参考用户指定的只读参考目录迁移并适配。

## 实现范围

目标为 2026 版；参考为 2025 版，不能整体覆盖。保留现有 staff `cs336-basics`、测试断言、项目依赖及提交脚本。

- `tests/adapters.py`：补全现有 8 个接口。
- `flash_pytorch.py`：迁移 online-softmax 分块前向，补齐因果掩码、非整块长度、FP16 的 FP32 累积；将反向改为逐块重算概率，避免完整 Q×K 中间矩阵。
- `flash_triton.py`：迁移参考前向/反向 kernel，修正前向尾块填充 key 进入 softmax 分母的问题。没有本机 CUDA，未声称 Triton 已通过设备编译或 GPU 验证。
- `ddp/ddp_overlap_individual_parameters.py`：迁移初始化广播、梯度就绪异步 all-reduce、反向后等待与平均。
- `ddp/shared_optimizer.py`：迁移按参数分配状态的优化器；修复闭包重复调用、参数组生成器、scheduler 超参数同步及保存/恢复。状态为每个 rank 的本地分片，未实现跨拓扑合并恢复。
- `ddp/fsdp.py`：新版新增要求。对 Linear/Embedding 实际分片 FP32 主权重，按层聚合计算权重，反向重聚合，异步 reduce-scatter 梯度；其他参数复制并平均梯度。支持共享权重与非整除填充。即时聚合，不预取未来层；全局只保留最近一次异步归约的完整通信缓冲，及时释放此前层的缓冲。稀疏 Embedding 和 max_norm 明确不支持。
- `benchmark.py`：沿用参考计时方法并接入本版 staff 模型，支持前向、前后向、训练步、DDP/FSDP、可选分片优化器、精度及编译选项。未复制依赖旧模型接口的参考实验脚本。
- `test/`：新增注意力边界、优化器 API、双 rank FSDP 共享/填充/缓冲生命周期回归。

## 实际验证

项目目录执行，macOS/Linux 与 Windows CMD 均可逐行运行：

```sh
uv run pytest -q
uv run test/edge_checks.py
uv run test/fsdp_edge_checks.py
uv run cs336_systems/benchmark.py --model-size tiny --batch-size 1 --context-length 8 --vocab-size 100 --warmup 1 --steps 2
uv run torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29601 cs336_systems/benchmark.py --model-size tiny --batch-size 1 --context-length 8 --vocab-size 100 --parallel fsdp --warmup 0 --steps 1
```

- 原始全套：**10 passed、4 skipped**。4 项均为缺 CUDA 的 Triton 测试；没有新增 skip。完整输出见 `pytest-final.txt`。
- 注意力 8 个额外用例：float32/float16、35×70/70×35 的矩形尾块、因果/非因果；输出及 Q/K/V 梯度与原生 autograd 公式一致。
- 优化器：闭包只执行一次、生成器参数组、动态学习率、保存时同步 scheduler 的最新学习率及加载恢复均通过。
- 两个真实 Gloo rank：共享的 11×7 Embedding/Linear 权重每 rank 39 个数，复制 bias/LayerNorm 梯度、两次 SGD 更新与全 batch 基线一致；所有层合计至多一个完整异步梯度缓冲，同步后无 pending。
- 单进程与双 rank FSDP benchmark CLI 的小型功能 smoke 通过。tiny 是合成小模型，不能用于手册标准模型或 GPU 性能结论。
- `torchrun --standalone` 在本机被 hostname/代理 DNS 解析影响，改用显式 `127.0.0.1` 的静态 rendezvous 后通过；未改系统代理配置。
- 新 FSDP、benchmark、边界检查的 Ruff 及所有实现的 Python 编译检查通过。

没有运行真实数据训练、GPU 性能实验、Nsight、编译加速实验、排行榜或外部提交。计时 smoke 只证明入口能运行，不证明性能提升。算法 TODO 已接通；数学推导和依赖硬件的实测结果没有编造。

是否属于重要实验：否。实验日志：不需要（本次为实现及功能验证）。未执行 Git 提交或推送。


## 建议执行的 Git 命令

在本作业目录执行。以下仅为建议，本次没有执行暂存、提交或推送；未纳入无关 `.DS_Store` 文件。

```sh
git status --short
git diff
git add -- cs336_systems/benchmark.py cs336_systems/ddp/__init__.py cs336_systems/ddp/ddp_overlap_individual_parameters.py cs336_systems/ddp/fsdp.py cs336_systems/ddp/shared_optimizer.py cs336_systems/flash_pytorch.py cs336_systems/flash_triton.py doc/implementation.md doc/pytest-final.txt test/edge_checks.py test/fsdp_edge_checks.py tests/adapters.py
git diff --cached
git commit -m 'feat: implement assignment2-systems TODOs'
```
