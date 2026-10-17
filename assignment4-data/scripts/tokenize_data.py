"""Stream filtered JSONL to GPT-2 uint16 IDs with one EOS per document."""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def tokenize(inputs: list[Path], output: Path) -> int:
    # 由 Codex 实现 handout 的 GPT-2 序列化：逐文档写 uint16，避免整个语料驻留内存。
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    output.parent.mkdir(parents=True, exist_ok=True)
    tokens = 0
    with output.open('xb') as destination:
        for source in inputs:
            with source.open(encoding='utf-8') as documents:
                for line in documents:
                    text = json.loads(line)['text']
                    ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                    if max(ids) > np.iinfo(np.uint16).max:
                        raise ValueError('Tokenizer produced an ID exceeding uint16')
                    np.asarray(ids, dtype=np.uint16).tofile(destination)
                    tokens += len(ids)
    print(json.dumps({'tokens': tokens, 'bytes': output.stat().st_size, 'output': str(output)}))
    return tokens


def main() -> None:
    # 由 Codex 实现可直接执行的 tokenizer 入口。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='Filtered JSONL, one text field per document')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    tokenize(args.inputs, args.output)


if __name__ == '__main__':
    main()
