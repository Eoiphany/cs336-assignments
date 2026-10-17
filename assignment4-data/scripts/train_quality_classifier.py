"""Train fastText on independently collected labeled documents, never on test fixtures."""
import argparse
import json
import tempfile
from pathlib import Path

import fasttext


def train(positive: Path, negative: Path, output: Path, epochs: int = 5, threads: int = 4) -> None:
    # 由 Codex 实现 handout 的质量分类训练；输入 JSONL 每行含 text 字段。
    if epochs < 1 or threads < 1:
        raise ValueError('epochs and threads must be positive')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing quality model: {output}')
    counts = {'hq': 0, 'lq': 0}
    with tempfile.TemporaryDirectory() as directory:
        training_path = Path(directory) / 'train.txt'
        with training_path.open('w', encoding='utf-8') as stream:
            for label, source in [('hq', positive), ('lq', negative)]:
                with source.open(encoding='utf-8') as documents:
                    for line in documents:
                        text = ' '.join(json.loads(line)['text'].split())
                        if text:
                            stream.write(f'__label__{label} {text}\n')
                            counts[label] += 1
        if not all(counts.values()):
            raise ValueError('Both positive and negative corpora must contain nonempty documents.')
        model = fasttext.train_supervised(input=str(training_path), epoch=epochs, thread=threads,
                                         wordNgrams=2, dim=64, loss='softmax')
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(output))
    print(json.dumps({'examples': counts, 'model': str(output)}))


def main() -> None:
    # 由 Codex 实现可直接执行的质量分类器训练入口。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positive', type=Path, required=True, help='JSONL from Wikipedia-linked pages')
    parser.add_argument('--negative', type=Path, required=True, help='JSONL sampled independently from Common Crawl')
    parser.add_argument('--output', type=Path, default=Path('runs/quality_fasttext.bin'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    train(args.positive, args.negative, args.output, args.epochs, args.threads)


if __name__ == '__main__':
    main()
