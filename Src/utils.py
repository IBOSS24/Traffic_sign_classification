import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_NAMES_PATH = PROJECT_ROOT / "class_names.json"


def load_class_names(path=DEFAULT_CLASS_NAMES_PATH) -> list[str]:
    """Load and validate the traffic-sign class label list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Class names file was not found at {path}")

    with path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise ValueError("class_names.json must contain a JSON list of strings")

    return class_names


def top_k_predictions(prediction_scores, class_names: list[str], k: int = 5) -> list[tuple[str, float]]:
    """Return top-k class labels and probabilities from a model output vector."""
    scores = np.asarray(prediction_scores, dtype=np.float32).reshape(-1)

    if len(scores) != len(class_names):
        raise ValueError(
            f"Prediction length ({len(scores)}) does not match class count ({len(class_names)})"
        )

    k = min(k, len(scores))
    top_indices = np.argsort(scores)[-k:][::-1]

    return [(class_names[int(index)], float(scores[int(index)])) for index in top_indices]
