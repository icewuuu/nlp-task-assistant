from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict:
    """Macierz pomyłek + ACC/ERR/P/R/F1 (też per klasa i średnie)."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    if total == 0:
        raise ValueError("Pusta macierz pomyłek.")

    correct = int(np.trace(cm))
    acc = correct / total
    err = 1.0 - acc

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, class_name in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        support = tp + fn
        per_class[class_name] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    # macro = zwykła średnia, weighted = z wagą support
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    support_sum = sum(supports) or 1
    weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / support_sum
    weighted_recall = sum(r * s for r, s in zip(recalls, supports)) / support_sum
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / support_sum

    return {
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "ACC": acc,
        "ERR": err,
        "per_class": per_class,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "weighted": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
    }


def format_confusion_matrix(metrics: dict) -> str:
    """Macierz + metryki jako tekst (pod telegrama)."""
    labels = metrics["labels"]
    cm = metrics["confusion_matrix"]

    header = "true\\pred".ljust(14) + "".join(f"{lab[:10]:>12}" for lab in labels)
    lines = [header]
    for i, lab in enumerate(labels):
        row = f"{lab[:12]:<14}" + "".join(f"{cm[i][j]:>12}" for j in range(len(labels)))
        lines.append(row)

    summary = (
        f"\nACC = {metrics['ACC']:.3f} | ERR = {metrics['ERR']:.3f} | "
        f"F1 macro = {metrics['macro']['f1']:.3f} | "
        f"F1 weighted = {metrics['weighted']['f1']:.3f}"
    )

    lines.append("\nPer class (one-vs-rest):")
    for name, m in metrics["per_class"].items():
        lines.append(
            f"  {name}: P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} (n={m['support']})"
        )

    return "\n".join(lines) + summary


def format_all_matrices(detailed: list[dict]) -> str:
    """Wszystkie macierze jedna pod drugą."""
    if not detailed:
        return "Brak macierzy."

    blocks = ["Macierze pomyłek (6 konfiguracji)", "=" * 40]
    for item in detailed:
        blocks.append("")
        blocks.append(f"{item['repr']} + {item['method']}")
        blocks.append(format_confusion_matrix(item["metrics"]))
        blocks.append("-" * 40)
    return "\n".join(blocks)


def format_comparison_table(results: list[dict]) -> str:
    """Prosta tabela ACC/ERR/F1."""
    if not results:
        return "Brak wyników."

    lines = [
        "Porównanie:",
        f"{'repr':<12}{'method':<10}{'ACC':>8}{'ERR':>8}{'F1_mac':>10}",
        "-" * 48,
    ]
    best = max(results, key=lambda r: r["F1_macro"])
    for r in results:
        mark = " *" if (
            r["repr"] == best["repr"] and r["method"] == best["method"]
        ) else ""
        lines.append(
            f"{r['repr']:<12}{r['method']:<10}"
            f"{r['ACC']:>8.3f}{r['ERR']:>8.3f}{r['F1_macro']:>10.3f}{mark}"
        )

    lines.append("")
    lines.append(
        f"Najlepsza: {best['repr']} + {best['method']} "
        f"(F1={best['F1_macro']:.3f}, ACC={best['ACC']:.3f})"
    )
    lines.append(
        "Komentarz: wyższe F1/ACC = lepsza para repr+klasyfikator na tym zbiorze."
    )
    return "\n".join(lines)
