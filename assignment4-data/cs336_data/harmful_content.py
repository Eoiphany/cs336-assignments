from cs336_data.classifiers import predict


def _classify(text: str, kind: str, positive: str, negative: str) -> tuple[str, float]:
    # 由 Codex 迁移参考实现并修复：未知标签必须显式报错，不能默认当作阴性。
    if not text.strip():
        return negative, 0.0
    raw, confidence = predict(kind, text)[0]
    normalized = raw.lower().replace('_', '-').replace(' ', '-')
    positives = {positive, '1', 'pos', 'positive', 'true'}
    negatives = {negative, '0', 'neg', 'negative', 'false'}
    if normalized in positives:
        return positive, confidence
    if normalized in negatives:
        return negative, confidence
    raise ValueError(f'Unsupported {kind} model label: {raw!r}')


def classify_nsfw(text: str) -> tuple[str, float]:
    # 由 Codex 迁移参考实现并修复 Dolma 模型加载路径。
    return _classify(text, 'nsfw', 'nsfw', 'non-nsfw')


def classify_toxic_speech(text: str) -> tuple[str, float]:
    # 由 Codex 迁移参考实现并修复 Dolma 模型加载路径。
    return _classify(text, 'toxic', 'toxic', 'non-toxic')
