"""
Evaluation metrics, plotting, and analysis helpers for NLP experiments.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Union


def compute_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[List[Any]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes standard classification metrics (accuracy, precision, recall, F1, and full report).

    Args:
        y_true: Ground truth target values.
        y_pred: Estimated targets returned by a classifier.
        labels: Optional list of label indices/values.
        target_names: Optional display names matching labels.

    Returns:
        Dict[str, Any]: Dictionary containing metric scores and text report.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report_text": report,
    }


def plot_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[List[Any]] = None,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: Optional[str] = None,
    figsize: tuple = (6, 5),
) -> Any:
    """Plots and optionally saves a styled confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Explicit list of class labels.
        class_names: Display names for tick labels.
        normalize: If True, normalize matrix by true class totals.
        title: Title of the figure.
        cmap: Matplotlib colormap.
        save_path: Optional file path to save the generated image.
        figsize: Figure dimensions (width, height).

    Returns:
        plt.Figure: The generated Matplotlib figure.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fmt = ".2f" if normalize else "d"

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    tick_labels = class_names if class_names is not None else (labels if labels is not None else "auto")

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="semibold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def compute_perplexity(
    cross_entropy_loss: Optional[float] = None,
    log_likelihood: Optional[float] = None,
    num_tokens: Optional[int] = None,
) -> float:
    """Calculates language model perplexity.

    Args:
        cross_entropy_loss (float, optional): Average cross-entropy loss (base e).
        log_likelihood (float, optional): Total log likelihood of the corpus (base e or 2).
        num_tokens (int, optional): Total number of evaluated tokens.

    Returns:
        float: Perplexity score.
    """
    if cross_entropy_loss is not None:
        return math.exp(cross_entropy_loss)

    if log_likelihood is not None and num_tokens is not None and num_tokens > 0:
        avg_neg_log_lik = -log_likelihood / num_tokens
        return math.exp(avg_neg_log_lik)

    raise ValueError("Provide either cross_entropy_loss or both (log_likelihood, num_tokens).")


def compute_token_tagging_accuracy(
    true_sequences: List[List[str]], pred_sequences: List[List[str]]
) -> Dict[str, float]:
    """Computes token-level accuracy and sequence accuracy for sequence tagging (POS, NER).

    Args:
        true_sequences: List of ground-truth tag sequences.
        pred_sequences: List of predicted tag sequences.

    Returns:
        Dict[str, float]: token_accuracy and exact_sequence_accuracy.
    """
    total_tokens = 0
    correct_tokens = 0
    total_seqs = len(true_sequences)
    correct_seqs = 0

    for true_seq, pred_seq in zip(true_sequences, pred_sequences):
        seq_correct = True
        for t, p in zip(true_seq, pred_seq):
            total_tokens += 1
            if t == p:
                correct_tokens += 1
            else:
                seq_correct = False
        if len(true_seq) != len(pred_seq):
            seq_correct = False
        if seq_correct:
            correct_seqs += 1

    token_acc = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0
    seq_acc = (correct_seqs / total_seqs) if total_seqs > 0 else 0.0

    return {
        "token_accuracy": token_acc,
        "exact_sequence_accuracy": seq_acc,
        "total_tokens": total_tokens,
        "total_sequences": total_seqs,
    }
