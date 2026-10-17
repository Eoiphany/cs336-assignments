import hashlib
import os
from pathlib import Path
from typing import Dict


def _line_key(line: str) -> bytes:
    # 由 Codex 迁移参考实现并修复。
    """
    Compute a stable hash key for a line.

    We normalize only the training newline characters so that lines with/without
    final newline are treated as the same "line content".
    """
    # Strip only new line characters (keep other whitespace intact).
    normalized = line.rstrip("\n").rstrip("\r")
    # 128-bit digest is plenty to avoid collisions in tests.
    return hashlib.blake2b(normalized.encode("utf-8", errors="replace"), digest_size=16).digest()


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike) -> None:
    # 由 Codex 迁移参考实现并修复。
    """
    Perform exact line-level deduplication across a collection of the input files.

    - First pass: count occurrences of each line (by hash) across all files.
    - Second pass: rewrite each file to output_directory, keeping only lines whose
      global count is exactly 1.

    Output filenames must match input filenames.
    """
    in_paths = [Path(p) for p in input_files]
    # 由 Codex 修复：扁平输出不能无声覆盖同名输入或原有处理结果。
    if len({p.name for p in in_paths}) != len(in_paths):
        raise ValueError('Input basenames must be unique in the flat output directory.')
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {out_dir}')

    # Pass 1: global line frequency (by hash)
    counts: Dict[bytes, int] = {}
    for p in in_paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                k = _line_key(line)
                counts[k] = counts.get(k, 0) + 1

    # Pass 2: rewrite files, keeping only globally-unique lines
    for p in in_paths:
        out_path = out_dir / p.name
        with open(p, "r", encoding="utf-8", errors="replace") as fin, open(
            out_path, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if counts.get(_line_key(line), 0) == 1:
                    # Write the originali line exactly (preserve original newline if present)
                    fout.write(line)
