import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def _to_binary_latent_dataset(
    dataloader: Any,
    class_0: str,
    class_1: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract latent vectors and build binary labels for the selected class pair.
    """
    dataset = dataloader.dataset
    idx_to_name = dataset.unique_classes

    xs: List[np.ndarray] = []
    ys: List[int] = []

    for latent_batch, label_batch in dataloader:
        latent_np = latent_batch.detach().cpu().numpy()
        label_np = label_batch.detach().cpu().numpy()

        for latent_vec, label_idx in zip(latent_np, label_np):
            class_name = idx_to_name[int(label_idx)]
            if class_name == class_0:
                xs.append(latent_vec.astype(np.float32))
                ys.append(0)
            elif class_name == class_1:
                xs.append(latent_vec.astype(np.float32))
                ys.append(1)

    if not xs:
        raise ValueError(
            f"No samples found for class pair ('{class_0}', '{class_1}') in this split."
        )

    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    return x, y


def _fit_projection_from_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build a 1D latent axis from class means and project train samples onto it.
    """
    mu0 = x_train[y_train == 0].mean(axis=0)
    mu1 = x_train[y_train == 1].mean(axis=0)

    direction = mu1 - mu0
    direction_norm = np.linalg.norm(direction)
    if direction_norm < eps:
        raise ValueError("Class means are too close; cannot build a stable projection axis.")

    direction = direction / direction_norm
    origin = 0.5 * (mu0 + mu1)

    x_proj_train = ((x_train - origin) @ direction).reshape(-1, 1)
    return x_proj_train, direction, origin, direction_norm


def _project(x: np.ndarray, direction: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return ((x - origin) @ direction).reshape(-1, 1)


def _compute_metrics(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)

    # Point-wise slope magnitude dp/dx = |a| * p * (1-p), where a is the logistic slope.
    slope_per_sample = abs(float(model.coef_[0, 0])) * proba * (1.0 - proba)

    metrics: Dict[str, float] = {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "mean_probability": float(np.mean(proba)),
        "mean_local_slope": float(np.mean(slope_per_sample)),
        "max_local_slope": float(np.max(slope_per_sample)),
    }

    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 2-parameter logistic regression on latent vectors and inspect slope-based uncertainty."
    )
    parser.add_argument("--data_dir", type=str, default="ade20k_data/ADEData2016")
    parser.add_argument("--latent_dir", type=str, required=True, help="Directory with train/validation/test latent files")
    parser.add_argument("--class_0", type=str, default=None, help="Negative class name")
    parser.add_argument("--class_1", type=str, default=None, help="Positive class name")
    parser.add_argument("--n_common_labels", type=int, default=None)
    parser.add_argument("--exclude_concepts", type=str, nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--regularization_C", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save results as JSON")
    args = parser.parse_args()

    # Lazy import keeps CLI help responsive even when torch environment is unavailable.
    from ade20k_dataset import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        train_augmentation=False,
        pin_memory=False,
        n_common_labels=args.n_common_labels,
        exclude_concepts=args.exclude_concepts,
        latent_dir=args.latent_dir,
    )

    class_0 = args.class_0
    class_1 = args.class_1

    if class_0 is None and class_1 is None and args.n_common_labels == 2:
        available_classes = list(train_loader.dataset.unique_classes)
        if len(available_classes) != 2:
            raise ValueError(
                "Automatic class selection requires exactly 2 available classes, "
                f"but found {len(available_classes)}: {available_classes}"
            )
        class_0, class_1 = available_classes[0], available_classes[1]
        print(f"Auto-selected classes from n_common_labels=2: class_0='{class_0}', class_1='{class_1}'")
    elif (class_0 is None) != (class_1 is None):
        raise ValueError("Provide both --class_0 and --class_1, or neither.")
    elif class_0 is None and class_1 is None:
        raise ValueError(
            "Missing class names. Provide --class_0 and --class_1, "
            "or set --n_common_labels 2 to auto-select the two labels."
        )

    x_train, y_train = _to_binary_latent_dataset(train_loader, class_0, class_1)
    x_val, y_val = _to_binary_latent_dataset(val_loader, class_0, class_1)
    x_test, y_test = _to_binary_latent_dataset(test_loader, class_0, class_1)

    x_proj_train, direction, origin, direction_norm = _fit_projection_from_train(
        x_train=x_train,
        y_train=y_train,
        eps=args.eps,
    )
    x_proj_val = _project(x_val, direction, origin)
    x_proj_test = _project(x_test, direction, origin)

    # 2-parameter logistic model: p(y=1|x) = sigmoid(a*x + b)
    model = LogisticRegression(
        penalty="l2",
        C=args.regularization_C,
        max_iter=args.max_iter,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(x_proj_train, y_train)

    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    midpoint = float(-intercept / slope) if abs(slope) > args.eps else float("inf")
    max_slope = abs(slope) / 4.0

    train_metrics = _compute_metrics(model, x_proj_train, y_train)
    val_metrics = _compute_metrics(model, x_proj_val, y_val)
    test_metrics = _compute_metrics(model, x_proj_test, y_test)

    results = {
        "class_pair": [class_0, class_1],
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "projection": {
            "direction_norm_before_normalization": float(direction_norm),
            "origin_shape": list(origin.shape),
        },
        "logistic_parameters": {
            "slope_a": slope,
            "intercept_b": intercept,
            "decision_midpoint_x0": midpoint,
            "max_theoretical_slope_abs_a_over_4": max_slope,
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        "interpretation": {
            "notes": [
                "Higher |slope_a| means steeper transition and less uncertainty near boundary.",
                "Lower |slope_a| means softer transition and higher uncertainty spread.",
                "Max local slope occurs at p=0.5 and equals |slope_a|/4.",
            ]
        },
    }

    print("\n=== 2-Parameter Logistic Regression On Latent Space ===")
    print(f"Class 0: {class_0}")
    print(f"Class 1: {class_1}")
    print(f"Samples (train/val/test): {len(y_train)} / {len(y_val)} / {len(y_test)}")
    print(f"slope a: {slope:.6f}")
    print(f"intercept b: {intercept:.6f}")
    print(f"midpoint x0 = -b/a: {midpoint:.6f}")
    print(f"max local slope |a|/4: {max_slope:.6f}")

    print("\nMetrics:")
    print(json.dumps(results["metrics"], indent=2))

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    main()