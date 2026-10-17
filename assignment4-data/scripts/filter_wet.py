"""Filter WET conversion records in parallel, saving JSONL and stage counts."""
import argparse
import gzip
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from warcio.archiveiterator import ArchiveIterator

from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import is_english
from cs336_data.pii_masking import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality_classifier import quality_score
from cs336_data.quality_rules import gopher_quality_filter


def filter_file(task: tuple) -> dict:
    # 由 Codex 实现 handout 的 WET 过滤流水线，每阶段统计保留和丢弃数。
    source, output, check_language, check_harm, min_quality, harmful_threshold = task
    counts = Counter()
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rb') as data, output.open('x', encoding='utf-8') as destination:
        for record in ArchiveIterator(data):
            if record.rec_type != 'conversion':
                continue
            counts['input'] += 1
            text = record.content_stream().read().decode('utf-8', errors='replace')
            if not text.strip():
                counts['discard_empty'] += 1
                continue
            counts['after_empty'] += 1
            if check_language and not is_english(text):
                counts['discard_language'] += 1
                continue
            counts['after_language'] += 1
            if check_harm:
                nsfw, nsfw_confidence = classify_nsfw(text)
                toxic, toxic_confidence = classify_toxic_speech(text)
                if (nsfw == 'nsfw' and nsfw_confidence >= harmful_threshold) or (
                    toxic == 'toxic' and toxic_confidence >= harmful_threshold
                ):
                    counts['discard_harmful'] += 1
                    continue
            counts['after_harmful'] += 1
            if not gopher_quality_filter(text):
                counts['discard_gopher'] += 1
                continue
            counts['after_gopher'] += 1
            if min_quality is not None and quality_score(text) < min_quality:
                counts['discard_quality'] += 1
                continue
            counts['after_quality'] += 1
            for mask in (mask_emails, mask_phone_numbers, mask_ips):
                text, masked = mask(text)
                counts['pii_masked'] += masked
            destination.write(json.dumps({'text': text, 'url': record.rec_headers.get_header('WARC-Target-URI')}, ensure_ascii=False) + '\n')
            counts['kept'] += 1
    return dict(counts)


def main() -> None:
    # 由 Codex 实现并行入口：不启动 LM 训练，独立输出真实过滤统计。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--check-language', action='store_true', help='Not needed for already English-filtered WETs')
    parser.add_argument('--check-harmful', action='store_true', help='Requires both Dolma fastText models')
    parser.add_argument('--min-quality', type=float, help='Optional high-quality probability; requires trained quality model')
    parser.add_argument('--harmful-threshold', type=float, default=0.8)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.harmful_threshold <= 1 or (args.min_quality is not None and not 0 <= args.min_quality <= 1):
        parser.error('workers must be positive and probability thresholds must be in [0, 1]')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tasks = [(source, args.output_dir / f'{i:05d}.jsonl', args.check_language, args.check_harmful,
              args.min_quality, args.harmful_threshold) for i, source in enumerate(args.inputs)]
    start = time.monotonic()
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(filter_file, tasks):
            counts.update(result)
    discarded = counts['input'] - counts['kept']
    report = {'counts': dict(counts), 'seconds': time.monotonic() - start,
              'discard_proportions': {key: value / discarded if discarded else 0.0
                                      for key, value in counts.items() if key.startswith('discard_')}}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
