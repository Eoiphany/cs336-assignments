from cs336_data.classifiers import predict


def identify_language(text: str) -> tuple[str, float]:
    # 由 Codex 迁移参考实现并修复模型路径和空白输入处理。
    if not text.strip():
        return 'unknown', 0.0
    language, score = predict('language', text)[0]
    return ('zh' if language.startswith('zh') else language), score


def is_english(text: str, threshold: float = 0.7) -> bool:
    # 由 Codex 实现 handout 要求的 English WET 概率阈值。
    language, confidence = identify_language(text)
    return language == 'en' and confidence >= threshold
