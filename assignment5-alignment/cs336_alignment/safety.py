"""Optional alignment handout primitives: packed SFT data, parsing, and DPO.

Dataset input is JSONL with prompt/response fields; batches contain shifted token
IDs (B,T). DPO takes a chosen/rejected pair and returns a differentiable scalar.
No downloads or model training are performed by this module.
Validation on macOS / Windows CMD: uv run pytest tests/test_data.py tests/test_metrics.py tests/test_dpo.py.
"""
import json
from pathlib import Path
import random
import re

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


_TEMPLATE = (Path(__file__).parent / 'prompts_safety' / 'alpaca_sft.prompt').read_text()


class PackedSFTDataset(Dataset):
    """Pack tokenized instruction documents into fixed-length shifted examples."""

    def __init__(self, tokenizer, dataset_path, seq_length: int, shuffle: bool):
        # 由 Codex 根据可选 safety handout 修复并实现。
        if seq_length < 1:
            raise ValueError('seq_length must be positive')
        with open(dataset_path, encoding='utf-8') as handle:
            examples = [json.loads(line) for line in handle if line.strip()]
        if shuffle:
            random.shuffle(examples)
        ids = []
        for example in examples:
            text = _TEMPLATE.format(instruction=example['prompt'], response=example['response']).rstrip()
            ids.extend(tokenizer.encode(text))
            ids.append(tokenizer.eos_token_id)
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        # 由 Codex 实现；最后一个输入块必须还存在一个 next-token 标签。
        return max(0, (len(self.tokens) - 1) // self.seq_length)

    def __getitem__(self, index):
        # 由 Codex 实现；相邻输入块不重叠，标签整体右移一个 token。
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.seq_length
        return {'input_ids': self.tokens[start:start + self.seq_length],
                'labels': self.tokens[start + 1:start + self.seq_length + 1]}


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    """Iterate over one complete dataset epoch, retaining the final partial batch."""
    # 由 Codex 根据目标文档修复并实现。
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_mmlu_response(mmlu_example: dict, model_output: str):
    """Parse an explicit option letter or an unambiguous complete option string."""
    # 由 Codex 根据目标文档修复并实现；数字选项使用词边界避免 100 匹配 10000。
    letter = re.search(r'(?:^\s*|(?:answer|option|choice)\s*(?:is|:)\s*)\(?([A-D])\)?(?:[.\s:]|$)',
                       model_output, flags=re.IGNORECASE)
    if letter:
        return letter.group(1).upper()
    matches = [chr(65 + index) for index, option in enumerate(mmlu_example['options'])
               if re.search(r'(?<!\w)' + re.escape(option) + r'(?!\w)', model_output, re.IGNORECASE)]
    return matches[0] if len(matches) == 1 else None


def parse_gsm8k_response(model_output: str):
    """Return the last generated number, removing thousands separators, or None."""
    # 由 Codex 根据目标文档修复并实现。
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', model_output)
    return numbers[-1].replace(',', '') if numbers else None


def compute_per_instance_dpo_loss(lm, lm_ref, tokenizer, beta: float,
                                   prompt: str, response_chosen: str, response_rejected: str):
    """Negative log-sigmoid of beta times policy/reference preference log-odds."""
    # 由 Codex 根据 safety handout 实现；参考模型无梯度并可位于不同设备。
    log_probs = []
    for model in (lm, lm_ref):
        device = next(model.parameters()).device
        scores = []
        for response in (response_chosen, response_rejected):
            text = _TEMPLATE.format(instruction=prompt, response=response) + tokenizer.eos_token
            ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
            with torch.set_grad_enabled(model is lm):
                logits = model(input_ids=ids[:, :-1]).logits
                score = F.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None]).sum()
            scores.append(score)
        log_probs.append(scores[0] - scores[1])
    return -F.logsigmoid(beta * (log_probs[0] - log_probs[1].to(log_probs[0].device)))
