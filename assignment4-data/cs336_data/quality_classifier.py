from cs336_data.classifiers import predict


def quality_score(text: str) -> float:
    # 由 Codex 迁移参考实现：返回训练模型给出的高质量概率，不用启发式冒充预测。
    if not text.strip():
        return 0.0
    probabilities = dict(predict('quality', text, k=-1))
    if 'hq' not in probabilities:
        raise ValueError('Quality model must be trained with __label__hq and __label__lq labels.')
    return probabilities['hq']


def classify_quality(text: str, threshold: float = 0.5) -> tuple[str, float]:
    # 由 Codex 迁移参考实现并修复：返回所预测标签的置信度，而非始终返回高质量概率。
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1]')
    score = quality_score(text)
    return ('wiki', score) if score >= threshold else ('cc', 1.0 - score)
