"""MinHash/LSH candidate generation followed by exact Jaccard clustering."""
import os
import random
import re
import shutil
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import mmh3


def _normalize_text(text: str) -> str:
    # 由 Codex 迁移参考实现：NFD、去重音、小写、移除标点、合并空白。
    decomposed = unicodedata.normalize('NFD', text).lower()
    # 由 Codex 修复：手册要求删除标点，附着标点不能人为拆出新单词。
    characters = (c for c in decomposed
                  if unicodedata.category(c) != 'Mn' and not unicodedata.category(c).startswith('P'))
    return re.sub(r'\s+', ' ', ''.join(characters)).strip()


def _word_ngrams(text: str, n: int) -> set[str]:
    # 由 Codex 修复并实现：短文档使用全文作为 shingle，避免不同短文都变为空集合。
    words = text.split()
    if len(words) < n:
        return {text}
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def minhash_deduplication(input_files: list[os.PathLike], num_hashes: int, num_bands: int,
                          ngrams: int, jaccard_threshold: float, output_directory: os.PathLike) -> None:
    # 由 Codex 迁移参考结构并修复：独立 MurmurHash、短文处理、参数验证和随机簇代表。
    if min(num_hashes, num_bands, ngrams) <= 0 or num_hashes % num_bands:
        raise ValueError('Hashes, bands and ngrams must be positive; hashes must be divisible by bands.')
    if not 0 <= jaccard_threshold <= 1:
        raise ValueError('jaccard_threshold must be in [0, 1]')
    paths = [Path(path) for path in input_files]
    if len({path.name for path in paths}) != len(paths):
        raise ValueError('Input basenames must be unique in the flat output directory.')
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    # 由 Codex 修复：拒绝残留旧代表文件的目录，避免输出中仍含本轮淘汰的文档。
    if any(destination.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {destination}')
    sets = [_word_ngrams(_normalize_text(path.read_text(encoding='utf-8')), ngrams) for path in paths]
    signatures = [tuple(min(mmh3.hash64(gram, seed, signed=False)[0] for gram in grams)
                        for seed in range(num_hashes)) for grams in sets]
    rows = num_hashes // num_bands
    buckets = defaultdict(list)
    for index, signature in enumerate(signatures):
        for band in range(num_bands):
            buckets[(band, signature[band * rows:(band + 1) * rows])].append(index)
    parents = list(range(len(paths)))

    def find(index: int) -> int:
        # 由 Codex 迁移参考实现：路径压缩用于传递重复簇。
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    candidates = {pair for indices in buckets.values() for pair in combinations(indices, 2)}
    for left, right in candidates:
        similarity = len(sets[left] & sets[right]) / len(sets[left] | sets[right])
        if similarity >= jaccard_threshold:
            parents[find(right)] = find(left)
    clusters = defaultdict(list)
    for index in range(len(paths)):
        clusters[find(index)].append(index)
    rng = random.Random(336)
    for members in clusters.values():
        source = paths[rng.choice(members)]
        output = destination / source.name
        if source.resolve() == output.resolve():
            raise ValueError('Output directory must differ from input directory.')
        shutil.copyfile(source, output)
