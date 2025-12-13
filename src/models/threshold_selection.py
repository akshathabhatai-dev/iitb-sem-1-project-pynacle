import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score

def optimize_balanced_thresholds(model, X_val, y_val, class_names, output_dir="balanced_thresholds"):
    """
    Computes per-class thresholds that BALANCE Precision, Recall, and F1.
    
    Balanced Score = sum of absolute differences between metrics.
    
    Returns:
        thresholds_dict – {class_name: best_threshold}
    """

    os.makedirs(output_dir, exist_ok=True)
    prob = model.predict_proba(X_val)
    num_classes = prob.shape[1]

    thresholds = np.linspace(0.01, 0.99, 200)
    threshold_dict = {}

    for class_id in range(num_classes):

        precision_list, recall_list, f1_list, balance_list = [], [], [], []

        for t in thresholds:
            y_pred = []

            for i in range(len(prob)):
                # Apply class-specific threshold
                if prob[i][class_id] >= t:
                    y_pred.append(class_id)
                else:
                    y_pred.append(np.argmax(prob[i]))

            precision = precision_score(y_val, y_pred, average=None, zero_division=0)[class_id]
            recall = recall_score(y_val, y_pred, average=None, zero_division=0)[class_id]
            f1 = f1_score(y_val, y_pred, average=None, zero_division=0)[class_id]

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

            # BALANCE SCORE (lower means metrics are closer)
            balance_err = abs(precision - recall) + abs(precision - f1) + abs(recall - f1)
            balance_list.append(balance_err)

        df = pd.DataFrame({
            "threshold": thresholds,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list,
            "balance_err": balance_list
        })

        # Find best balanced threshold (minimum imbalance)
        best_row = df.iloc[df["balance_err"].idxmin()]
        best_threshold = float(best_row["threshold"])

        # threshold_dict[class_names[class_id]] = best_threshold
        threshold_dict[class_id] = best_threshold

        # Plot metrics + balance error
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision_list, label="Precision")
        plt.plot(thresholds, recall_list, label="Recall")
        plt.plot(thresholds, f1_list, label="F1")
        plt.plot(thresholds, balance_list, label="Balance Error", linestyle='--')
        plt.title(f"Balanced Threshold Optimization - {class_names[class_id]}")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"class_{class_names[class_id]}_balanced.png"))
        plt.close()

    return threshold_dict
