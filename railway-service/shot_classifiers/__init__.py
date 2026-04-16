from .base import BaseShotClassifier, ClassificationResult
from .forehand import ForehandClassifier
from .backhand import BackhandClassifier
from .serve import ServeClassifier
from .slice import SliceClassifier

CLASSIFIERS: dict[str, BaseShotClassifier] = {
    "forehand": ForehandClassifier(),
    "backhand": BackhandClassifier(),
    "serve": ServeClassifier(),
    "slice": SliceClassifier(),
}


def classify_shot(frames: list[dict]) -> ClassificationResult:
    """Run all classifiers and return the highest-confidence result."""
    best: ClassificationResult | None = None
    for classifier in CLASSIFIERS.values():
        result = classifier.classify(frames)
        if result.confidence > 0 and (best is None or result.confidence > best.confidence):
            best = result
    if best is None:
        return ClassificationResult(shot_type="unknown", confidence=0.0, is_clean=False, camera_angle="unknown")
    return best


__all__ = [
    "BaseShotClassifier",
    "ClassificationResult",
    "ForehandClassifier",
    "BackhandClassifier",
    "ServeClassifier",
    "SliceClassifier",
    "CLASSIFIERS",
    "classify_shot",
]
