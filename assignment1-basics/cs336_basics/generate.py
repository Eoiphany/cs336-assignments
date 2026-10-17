"""Codex适配参考解码：可配置检查点、tokenizer、prompt和正确nucleus采样。

macOS/Windows CMD相同：
  uv run python cs336_basics/generate.py --checkpoint runs/实验/参数/last.pt --vocab vocab.pkl --merges merges.pkl --prompt "Once upon a time"
纯state_dict best.pt额外提供--config该run/config.json；完整epoch/last检查点
自带模型配置。--device默认cpu，--max-new-tokens默认128，--temperature默认1，
--top-p默认1，--end-token默认<|endoftext|>，--seed默认42。温度0为greedy。
输入tokenizer采用本项目Tokenizer.from_files的pickle vocab/merges格式；
输出包含原prompt和新生成文本，不创建优化器，不执行训练或联网操作。
"""

import argparse
import math
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cs336_basics.config import build_model, config_from_dict, load_config
from cs336_basics.tokenizer import Tokenizer


def top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Codex修复参考实现：保留累计概率首次达到阈值的token，再归一化。

    输入为1D非负有限概率；输出同形状。选择的最小前缀总概率>=top_p，
    包含跨过阈值的token（旧参考的cum<=top_p会错误丢弃它）。
    """
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if probs.ndim != 1 or not torch.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
        raise ValueError("probs must be a nonempty 1D finite, nonnegative probability vector")
    normalized = probs / probs.sum()
    sorted_probs, indices = torch.sort(normalized, descending=True)
    before = torch.cat((torch.zeros_like(sorted_probs[:1]), torch.cumsum(sorted_probs, dim=0)[:-1]))
    kept = sorted_probs * (before < top_p)
    filtered = kept / kept.sum()
    return torch.zeros_like(probs).scatter(0, indices, filtered)


@torch.no_grad()
def generate(model, prompt_ids, *, end_token_id, max_new_tokens=128, temperature=1.0, top_p=1.0):
    """Codex适配：自回归滑窗采样，输入[T]，返回[T+新token数]且遇EOS停止。"""
    if prompt_ids.ndim != 1 or not prompt_ids.numel():
        raise ValueError("prompt_ids must be a nonempty 1D tensor")
    if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("max_new_tokens and temperature must be nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    device = next(model.parameters()).device
    output = prompt_ids.to(device=device, dtype=torch.long)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(output[-model.context_length :].unsqueeze(0))[0, -1]
            if temperature == 0:
                token = int(torch.argmax(logits))
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                token = int(torch.multinomial(top_p_sampling(probs, top_p), 1))
            output = torch.cat((output, output.new_tensor([token])))
            if token == end_token_id:
                break
        return output
    finally:
        model.train(was_training)


def main():
    """Codex实现：加载纯权重/完整检查点，编码prompt、采样并解码文本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--merges", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--end-token", default="<|endoftext|>")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    full = isinstance(payload, dict) and "model_state_dict" in payload
    if args.config:
        cfg = load_config(args.config)
    elif full and "config" in payload:
        cfg = config_from_dict(payload["config"])
    else:
        parser.error("Pure state_dict or legacy checkpoints require --config")
    cfg.data.device = args.device
    torch.manual_seed(args.seed)
    model = build_model(cfg)
    model.load_state_dict(payload["model_state_dict"] if full else payload)
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[args.end_token])
    if max(tokenizer.vocab) >= cfg.model.vocab_size:
        raise ValueError("Tokenizer contains token IDs outside the model vocabulary")
    prompt = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long)
    ids = generate(
        model,
        prompt,
        end_token_id=tokenizer.special_id[args.end_token],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(tokenizer.decode(ids.tolist()))


if __name__ == "__main__":
    main()
