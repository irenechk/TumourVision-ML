"""Export readable JSON descriptions of the saved TumorVision pickle files."""

import json
import os
import pickle

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(BASE_DIR, "pickle")
READABLE_DIR = os.path.join(PICKLE_DIR, "readable")


def to_jsonable(value):
    """Convert common NumPy and Python values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def describe_tree(tree):
    """Return a readable representation of a fitted decision tree's nodes."""
    attributes = (
        "children_left",
        "children_right",
        "feature",
        "threshold",
        "value",
    )
    return {
        "node_count": int(tree.node_count),
        "nodes": {
            name: to_jsonable(getattr(tree, name))
            for name in attributes
        },
    }


def describe_object(obj):
    """Describe estimator settings and learned values without replacing the pickle."""
    description = {
        "type": f"{type(obj).__module__}.{type(obj).__name__}",
    }

    if hasattr(obj, "get_params"):
        description["parameters"] = to_jsonable(obj.get_params(deep=False))

    if hasattr(obj, "steps"):
        description["steps"] = [
            {"name": name, "model": describe_object(step)}
            for name, step in obj.steps
        ]

    fitted_attributes = {}
    for name, value in vars(obj).items():
        if name.endswith("_") and not name.startswith("__"):
            if name == "tree_":
                fitted_attributes[name] = describe_tree(value)
            elif name == "estimators_":
                fitted_attributes[name] = [
                    describe_object(estimator) for estimator in value
                ]
            else:
                fitted_attributes[name] = to_jsonable(value)

    # KNN stores its learned training samples in private attributes.
    for name in ("_fit_X", "_y"):
        if hasattr(obj, name):
            fitted_attributes[name] = to_jsonable(getattr(obj, name))

    if fitted_attributes:
        description["fitted_attributes"] = fitted_attributes
    return description


def main():
    if not os.path.isdir(PICKLE_DIR):
        raise FileNotFoundError(f"Pickle directory not found: {PICKLE_DIR}")

    os.makedirs(READABLE_DIR, exist_ok=True)
    pickle_files = sorted(
        name for name in os.listdir(PICKLE_DIR) if name.endswith(".pkl")
    )
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in: {PICKLE_DIR}")

    for filename in pickle_files:
        pickle_path = os.path.join(PICKLE_DIR, filename)
        with open(pickle_path, "rb") as pickle_file:
            obj = pickle.load(pickle_file)

        description = (
            to_jsonable(obj)
            if filename == "model_metadata.pkl"
            else describe_object(obj)
        )
        json_path = os.path.join(
            READABLE_DIR, f"{os.path.splitext(filename)[0]}.json"
        )
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(description, json_file, indent=2, allow_nan=False)
            json_file.write("\n")
        print(f"Created {json_path}")


if __name__ == "__main__":
    main()
