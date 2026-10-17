"""Lazy fastText loading, compatible with the 2026 offline asset layout."""
import os
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_FILES = {
    'language': ('lid.176.bin',),
    'nsfw': ('dolma_fasttext_nsfw_jigsaw_model.bin', 'jigsaw_fasttext_bigrams_nsfw_final.bin'),
    'toxic': ('dolma_fasttext_hatespeech_jigsaw_model.bin', 'jigsaw_fasttext_bigrams_hatespeech_final.bin'),
    'quality': ('quality_fasttext.bin',),
}


def model_path(kind: str) -> Path:
    # 由 Codex 实现：兼容环境变量、2026 下载目录和参考实现的旧目录。
    variable = f'CS336_{kind.upper()}_MODEL_PATH'
    if configured := os.environ.get(variable):
        candidates = [Path(configured).expanduser()]
    elif kind == 'quality':
        candidates = [_ROOT / 'runs' / 'quality_fasttext.bin']
    else:
        directories = [Path('/shared-data/classifiers'), _ROOT / 'local-shared-data/classifiers', _ROOT / 'data/classifiers']
        candidates = [directory / name for directory in directories for name in _MODEL_FILES[kind]]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    locations = ', '.join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f'Missing {kind} fastText model. Searched: {locations}. '
        f'Set {variable} to the trained/downloaded .bin file. See doc/implementation.md.'
    )


@lru_cache(maxsize=4)
def _load_model(path: str):
    # 由 Codex 实现：按解析后的模型路径缓存，避免重复加载大模型。
    import fasttext
    return fasttext.load_model(path)


def predict(kind: str, text: str, k: int = 1) -> list[tuple[str, float]]:
    # 由 Codex 实现：fastText 的预测输入必须是单行，置信度限制在 [0, 1]。
    text = ' '.join(text.split())
    labels, scores = _load_model(str(model_path(kind))).predict(text, k=k)
    return [(label.removeprefix('__label__'), max(0.0, min(1.0, float(score)))) for label, score in zip(labels, scores)]
