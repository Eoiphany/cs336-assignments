"""由 Codex 实现的归一化/输出回归：uv run test/dedup_edge_checks.py。"""

from pathlib import Path
from tempfile import TemporaryDirectory

from cs336_data.exact_deduplication import exact_line_deduplication
from cs336_data.minhash_deduplication import _normalize_text, minhash_deduplication


def check():
    # 由 Codex 实现：附着标点和重音不同的文档应归入相同重复簇。
    assert _normalize_text("Café's re-sume") == _normalize_text('cafes resume')
    with TemporaryDirectory() as directory:
        root = Path(directory)
        a, b = root / 'a.txt', root / 'b.txt'
        a.write_text("Café's re-sume", encoding='utf-8')
        b.write_text('cafes resume', encoding='utf-8')
        output = root / 'fuzzy'
        minhash_deduplication([a, b], 20, 5, 1, 1.0, output)
        assert len(list(output.iterdir())) == 1
        for operation in (
            lambda: minhash_deduplication([a, b], 20, 5, 1, 1.0, output),
            lambda: exact_line_deduplication([a, b], output),
        ):
            try:
                operation()
            except FileExistsError:
                pass
            else:
                raise AssertionError('Nonempty output directory must be rejected')
    print('dedup punctuation/accents/stale-output: passed')


if __name__ == '__main__':
    check()
