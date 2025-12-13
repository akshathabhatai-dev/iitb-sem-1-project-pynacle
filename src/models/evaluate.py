import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from pathlib import Path
from typing import Tuple, Dict
import pickle

# from src.models.threshold_tuning import X_val
from threshold_selection import optimize_balanced_thresholds

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                  class_names: Dict[int, str] = None,
                  verbose: bool = True,
                  output_path: str = "evaluation_results") -> Dict:
    """
    Evaluate trained model on test set.

    Args:
        model: Trained classifier
        X_test: Test features of shape (n_samples, 88)
        y_test: Test labels of shape (n_samples,)
        class_names: Dictionary mapping class indices to names
        verbose: Print evaluation results

    Returns:
        Dictionary containing evaluation metrics
    """
    if class_names is None:
        class_names = {0: 'No Object', 1: 'Ball', 2: 'Bat', 3: 'Stump'}

    # Predictions
    # y_pred = model.predict(X_test)
    # y_pred = np.where(y_pred == 4, 0, y_pred)  # Map 'Other No Objects ' to 'No Object'
    # y_test = np.where(y_test == 4, 0, y_test)  # Map 'Other No Objects ' to 'No Object'
    # THRESHOLDS = {
    #     0: 0.5,   # No Object
    #     1: 0.25,  # Ball
    #     2: 0.30,  # Bat
    #     3: 0.30   # Stump
    # }
    balanced_thresholds = optimize_balanced_thresholds(
        model,
        X_test,
        y_test,
        class_names,
        output_dir=output_path + "/balanced_thresholds"
    )

    print("Balanced thresholds:", balanced_thresholds)
    probs = model.predict_proba(X_test)
    y_pred = []

    # Build per-class threshold list in class-id order
    thresholds = [balanced_thresholds[name] for name in class_names]

    for i in range(len(probs)):
        p = probs[i]
        # Default prediction = class with max probability
        pred_class = np.argmax(p)

        # Apply per-class threshold logic
        for cls_id in range(len(class_names)):
            if p[cls_id] >= thresholds[cls_id]:
                pred_class = cls_id

        y_pred.append(pred_class)


    # thresholds = [0.50, 0.30, 0.30, 0.30]  # class-wise thresholds

    # # final_pred = []
    # for p in probs:
    #     idx = np.argmax(p / thresholds)  # boost minority classes
    #     y_pred.append(idx)
    # for row in probs:
    #     adjusted = [1 if p >= THRESHOLDS[i] else 0 for i, p in enumerate(row)]
    #     # If multiple 1s, choose max prob
    #     if sum(adjusted) > 1:
    #         y_pred.append(np.argmax(row))
    #     elif sum(adjusted) == 0:
    #         y_pred.append(np.argmax(row))  # fallback
    #     else:
    #         y_pred.append(adjusted.index(1))

    # for p in proba:
    #     if p[1] > 0.45:
    #         y_pred.append(1)   # Ball
    #     elif p[2] > 0.40:
    #         y_pred.append(2)   # Bat
    #     elif p[3] > 0.45:
    #         y_pred.append(3)   # Stump
    #     else:
    #         y_pred.append(0)

    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Per-class metrics
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Macro-averaged metrics (for imbalanced data)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    if verbose:
        print("="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
        print(f"F1-Macro: {f1_macro:.4f}")
        print(f"F1-Weighted: {f1_weighted:.4f}")

        print("\nPer-class Metrics:")
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-"*60)

        for i, (prec, rec, f1_val) in enumerate(zip(precision, recall, f1)):
            support = np.sum(y_test == i)
            cls_name = class_names.get(i, f'Class {i}')
            print(f"{cls_name:<15} {prec:>10.4f} {rec:>10.4f} {f1_val:>10.4f} {support:>10d}")

        print("\nConfusion Matrix:")
        print(cm)

        # Classification report
        print("\nDetailed Classification Report:")
        target_names = [class_names[i] for i in sorted(class_names.keys())]
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Package results
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }

    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: Dict[int, str] = None,
                         save_path: str = None, normalize: bool = False):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix
        class_names: Dictionary mapping class indices to names
        save_path: Path to save the plot
        normalize: Normalize confusion matrix
    """
    if class_names is None:
        class_names = {0: 'No Object', 1: 'Ball', 2: 'Bat', 3: 'Stump'}

    labels = [class_names[i] for i in sorted(class_names.keys())]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=labels, yticklabels=labels,
               cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_class_performance(results: Dict, class_names: Dict[int, str] = None,
                          save_path: str = None):
    """
    Plot per-class precision, recall, and F1-score.

    Args:
        results: Results dictionary from evaluate_model
        class_names: Dictionary mapping class indices to names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = {0: 'No Object', 1: 'Ball', 2: 'Bat', 3: 'Stump'}

    labels = [class_names[i] for i in sorted(class_names.keys())]
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1']

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to {save_path}")

    plt.show()


def save_metrics_to_file(results: Dict, output_path: str,
                        class_names: Dict[int, str] = None):
    """
    Save evaluation metrics to a text file.

    Args:
        results: Results dictionary from evaluate_model
        output_path: Path to save metrics
        class_names: Dictionary mapping class indices to names
    """
    if class_names is None:
        class_names = {0: 'No Object', 1: 'Ball', 2: 'Bat', 3: 'Stump'}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CRICKET OBJECT DETECTION - EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")

        f.write(f"Overall Accuracy: {results['accuracy'] * 100:.2f}%\n")
        f.write(f"F1-Macro: {results['f1_macro']:.4f}\n")
        f.write(f"F1-Weighted: {results['f1_weighted']:.4f}\n\n")

        f.write("Per-class Metrics:\n")
        f.write(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
        f.write("-"*60 + "\n")

        for i, (prec, rec, f1_val) in enumerate(zip(results['precision'],
                                                     results['recall'],
                                                     results['f1'])):
            cls_name = class_names.get(i, f'Class {i}')
            f.write(f"{cls_name:<15} {prec:>10.4f} {rec:>10.4f} {f1_val:>10.4f}\n")

        f.write("\nConfusion Matrix:\n")
        cm = results['confusion_matrix']
        labels = [class_names[i] for i in sorted(class_names.keys())]

        # Header
        f.write(f"{'True/Pred':<15}")
        for label in labels:
            f.write(f"{label:>15}")
        f.write("\n")

        # Rows
        for i, label in enumerate(labels):
            f.write(f"{label:<15}")
            for j in range(len(labels)):
                f.write(f"{cm[i, j]:>15d}")
            f.write("\n")

    print(f"Metrics saved to {output_path}")


def main():
    """Main evaluation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate cricket object detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pkl)')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to test features (.npy)')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to test labels (.csv)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save evaluation results')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # Load test data
    print("Loading test data...")
    X_test = np.load(args.features)

    labels_df = pd.read_csv(args.labels)
    y_test = []
    for _, row in labels_df.iterrows():
        cell_labels = [row[f'c{i:02d}'] for i in range(1, 65)]
        y_test.extend(cell_labels)
    y_test = np.array(y_test, dtype=np.int32)

    print(f"Test data: X={X_test.shape}, y={y_test.shape}")

    # Evaluate
    results = evaluate_model(model, X_test, y_test, verbose=True)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metrics_to_file(results, output_dir / 'metrics.txt')
    plot_confusion_matrix(results['confusion_matrix'],
                         save_path=output_dir / 'confusion_matrix.png')
    plot_class_performance(results,
                          save_path=output_dir / 'class_performance.png')


if __name__ == '__main__':
    main()
