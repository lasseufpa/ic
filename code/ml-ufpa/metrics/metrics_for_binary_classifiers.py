"""
Threshold scanning with a linear SVM
------------------------------------

Main idea:
    A classifier produces a SCORE for each example.
    We choose a THRESHOLD for that score:
        score >= threshold  -> predict class 1
        score <  threshold  -> predict class 0

The model is trained only once. Then we change the threshold and
observe how TP, TN, FP, FN and the evaluation metrics change.

The ROC and Precision-Recall curves are built directly from the
threshold-scanning table.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    auc,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# 1. A SMALL TRAINING SET
# ============================================================
#
# One input feature makes the example easy to visualize.
# 0 = negative class
# 1 = positive class
#
# There are only 12 examples, so the threshold table stays small.

X = np.array([1, 2, 3, 5, 6, 8,
              2.5, 4, 4.5, 6.5, 7, 9]).reshape(-1, 1)

y = np.array([0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 1, 1])


# ============================================================
# 2. TRAIN A LINEAR SVM
# ============================================================
#
# The SVM is trained once.
#
# decision_function() gives one score for each example.
# Positive scores are on the class-1 side of the SVM boundary.
#
# IMPORTANT:
# The SVM's default decision threshold is 0.
# We will later try MANY other thresholds.

model = LinearSVC(C=1, random_state=42, max_iter=10000)
model.fit(X, y)

scores = model.decision_function(X)

print("SVM decision scores:")
print(np.round(scores, 3))


# ============================================================
# 3. SHOW THE MODEL SCORES
# ============================================================

plt.figure(figsize=(9, 4))

plt.scatter(X[y == 0], scores[y == 0], label="Class 0")
plt.scatter(X[y == 1], scores[y == 1], label="Class 1")

plt.axhline(0, linestyle="--", label="SVM default threshold = 0")

plt.xlabel("Input feature x")
plt.ylabel("SVM score")
plt.title("The linear SVM assigns a score to each example")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 4. CHOOSE THE THRESHOLDS
# ============================================================
#
# Each example has one score.
#
# When the threshold passes one of these scores, one prediction
# changes. Therefore, these are the interesting thresholds.
#
# We also add two thresholds outside the score range:
#   - above the largest score: nobody is predicted as class 1
#   - below the smallest score: everybody is predicted as class 1

# what is np.r_? It is a way to concatenate arrays in numpy.
# In this case, it is used to create an array of thresholds that includes the maximum score plus a small value, the unique scores sorted in descending order, and the minimum score minus a small value. This ensures that we cover all possible thresholds for the SVM scores.
thresholds = np.r_[
    scores.max() + 0.01,
    np.sort(np.unique(scores))[::-1],
    scores.min() - 0.01
]


# ============================================================
# 5. SCAN THROUGH THE THRESHOLDS
# ============================================================
#
# This is the central part of the example.
#
# For EACH threshold:
#   1. Make predictions.
#   2. Count TP, TN, FP and FN.
#   3. Calculate the metrics.
#
# Notice that the model is NOT retrained.
# Only the decision threshold changes.

rows = []

for threshold in thresholds:

    # Apply the threshold to the SVM scores.
    predictions = (scores >= threshold).astype(int)

    # Confusion-matrix counts.
    TP = np.sum((y == 1) & (predictions == 1))
    TN = np.sum((y == 0) & (predictions == 0))
    FP = np.sum((y == 0) & (predictions == 1))
    FN = np.sum((y == 1) & (predictions == 0))

    # Metrics.
    accuracy = (TP + TN) / len(y)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # f1 is the harmonic mean of precision and recall. It is a common metric for imbalanced datasets.
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    rows.append([
        threshold, TP, TN, FP, FN,
        accuracy, precision, recall, specificity, f1
    ])


# ============================================================
# 6. THE THRESHOLD-SCANNING TABLE
# ============================================================

columns = [
    "Threshold", "TP", "TN", "FP", "FN",
    "Accuracy", "Precision", "Recall", "Specificity", "F1"
]

results = pd.DataFrame(rows, columns=columns)

print("\nThreshold-scanning table:")
print(results.round(3).to_string(index=False))


# ============================================================
# 7. METRICS VS THRESHOLD
# ============================================================
#
# Each ROW of the table corresponds to ONE threshold.
#
# The curves below are simply the columns of the table plotted
# against the threshold.

plt.figure(figsize=(9, 5))

plt.plot(results["Threshold"], results["Recall"],
         marker="o", label="Recall / Sensitivity")

plt.plot(results["Threshold"], results["Specificity"],
         marker="o", label="Specificity")

plt.plot(results["Threshold"], results["Precision"],
         marker="o", label="Precision")

plt.plot(results["Threshold"], results["Accuracy"],
         marker="o", label="Accuracy")

plt.xlabel("Decision threshold")
plt.ylabel("Metric")
plt.title("Metrics change when the decision threshold changes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 8. BUILD THE ROC CURVE FROM THE TABLE
# ============================================================
#
# ROC uses:
#
#     x-axis = False Positive Rate = 1 - Specificity
#     y-axis = True Positive Rate  = Recall / Sensitivity
#
# Therefore, the ROC curve can be calculated directly from
# the threshold-scanning table.
#
# ------------------------------------------------------------
# ONE POINT OF THE ROC CURVE: worked example
# ------------------------------------------------------------
#
# Suppose one row of the table gives:
#
#     TP = 4
#     TN = 4
#     FP = 2
#     FN = 2
#
# Sensitivity (TPR):
#
#     TPR = TP / (TP + FN)
#         = 4 / (4 + 2)
#         = 4 / 6
#         = 0.667
#
# Specificity:
#
#     Specificity = TN / (TN + FP)
#                 = 4 / (4 + 2)
#                 = 4 / 6
#                 = 0.667
#
# False Positive Rate:
#
#     FPR = 1 - Specificity
#         = 1 - 0.667
#         = 0.333
#
# Therefore, this ONE ROW of the table gives ONE POINT
# on the ROC curve:
#
#                 (FPR, TPR) = (0.333, 0.667)
#
# Repeating this calculation for every threshold gives
# all the points needed to draw the ROC curve.

fpr = 1 - results["Specificity"]
tpr = results["Recall"]

# newer versions of numpy have a trapezoid function, but scikit-learn has a more robust implementation.
# roc_auc_from_table = np.trapezoid(tpr, fpr)
roc_auc_from_table = auc(fpr, tpr)

plt.figure(figsize=(6, 6))

plt.plot(fpr, tpr, marker="o",
         label=f"ROC (AUC = {roc_auc_from_table:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--",
         label="Random classifier")

plt.xlabel("False Positive Rate = 1 - Specificity")
plt.ylabel("True Positive Rate = Recall")
plt.title("ROC curve calculated from the threshold table")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 9. BUILD THE PRECISION-RECALL CURVE FROM THE TABLE
# ============================================================
#
# Precision-Recall uses:
#
#     x-axis = Recall
#     y-axis = Precision
#
# Again, these values already exist in the threshold table.

recall = results["Recall"]
precision = results["Precision"]

plt.figure(figsize=(6, 6))

plt.plot(recall, precision, marker="o",
         label="Precision-Recall curve")

plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")
plt.title("Precision-Recall curve calculated from the threshold table")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 10. CHECK THE ROC AUC USING SCIKIT-LEARN
# ============================================================
#
# This is useful for showing students that the same ROC AUC can
# be obtained using a standard library function.
#
# The important teaching point is that the curve itself comes
# from changing the threshold.

roc_auc_sklearn = roc_auc_score(y, scores)
average_precision = average_precision_score(y, scores)

print("\nAUC checks:")
print(f"ROC AUC (from threshold table): {roc_auc_from_table:.3f}")
print(f"ROC AUC (scikit-learn):         {roc_auc_sklearn:.3f}")
print(f"Average Precision (PR metric):  {average_precision:.3f}")


# ============================================================
# 11. MAIN OBSERVATIONS
# ============================================================

print("\nMain observations:")
print("1. Each row of the table corresponds to one decision threshold.")
print("2. The SVM is trained only once; changing the threshold does not retrain it.")
print("3. As the threshold decreases, Recall/Sensitivity increases.")
print("4. As the threshold decreases, Specificity decreases.")
print("5. Precision is not monotonic: it can go up and down.")
print("6. There is a trade-off between Sensitivity and Specificity.")
print("7. There is also a trade-off between Precision and Recall.")
print("8. The ROC curve is obtained from (1 - Specificity, Recall).")
print("9. The Precision-Recall curve is obtained from (Recall, Precision).")
print("10. Different models produce different scores, so their effective")
print("    threshold values and threshold-scanning tables are different.")


# ============================================================
# 10. COMPARE TWO CLASSIFIERS USING ROC AND PR CURVES
# ============================================================
#
# The small dataset above is useful for understanding exactly
# how the threshold-scanning table is constructed.
#
# For comparing classifiers, however, it is useful to have
# more examples. Here we create a larger synthetic dataset and
# train TWO different classifiers.
#
# Important idea:
#
#     Each classifier produces its own scores.
#     Therefore, each classifier has its own effective thresholds.
#
# We can then compare the classifiers using:
#
#     ROC curve: True Positive Rate vs False Positive Rate
#     PR curve:  Precision vs Recall
#
# ------------------------------------------------------------


# Create a larger dataset.
# We keep it small enough to inspect, but large enough to make
# smooth-looking ROC and PR curves.
X_large, y_large = make_classification(
    n_samples=100,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=2,
    class_sep=0.1,
    random_state=42
)

# Train classifier 1: SVM
svm = SVC(C=1.0, random_state=42, max_iter=10000, kernel="rbf")
svm.fit(X_large, y_large)
svm_scores = svm.decision_function(X_large)

# Train classifier 2: Logistic Regression
logistic = LogisticRegression(random_state=42)
logistic.fit(X_large, y_large)

# Logistic regression gives probabilities.
logistic_scores = logistic.predict_proba(X_large)[:, 1]


# ------------------------------------------------------------
# Calculate ROC curves
# ------------------------------------------------------------

svm_fpr, svm_tpr, _ = roc_curve(y_large, svm_scores)
log_fpr, log_tpr, _ = roc_curve(y_large, logistic_scores)


# ------------------------------------------------------------
# Calculate Precision-Recall curves
# ------------------------------------------------------------

svm_precision, svm_recall, _ = precision_recall_curve(
    y_large, svm_scores
)

log_precision, log_recall, _ = precision_recall_curve(
    y_large, logistic_scores
)


# ------------------------------------------------------------
# Plot the ROC curves
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    svm_fpr,
    svm_tpr,
    marker="o",
    label=f"SVM (AUC = {roc_auc_score(y_large, svm_scores):.3f})"
)

plt.plot(
    log_fpr,
    log_tpr,
    marker="o",
    label=f"Logistic Regression (AUC = {roc_auc_score(y_large, logistic_scores):.3f})"
)

plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves: comparing two classifiers")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Plot the Precision-Recall curves
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    svm_recall,
    svm_precision,
    marker="o",
    label=f"SVM (AP = {average_precision_score(y_large, svm_scores):.3f})"
)

plt.plot(
    log_recall,
    log_precision,
    marker="o",
    label=f"Logistic Regression (AP = {average_precision_score(y_large, logistic_scores):.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curves: comparing two classifiers")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Print the summary
# ------------------------------------------------------------

print("\nClassifier comparison")
print("---------------------")

print(
    f"SVM:          "
    f"ROC AUC = {roc_auc_score(y_large, svm_scores):.3f}, "
    f"Average Precision = {average_precision_score(y_large, svm_scores):.3f}"
)

print(
    f"Logistic Regression: "
    f"ROC AUC = {roc_auc_score(y_large, logistic_scores):.3f}, "
    f"Average Precision = {average_precision_score(y_large, logistic_scores):.3f}"
)

# ------------------------------------------------------------
# Key observation for students
# ------------------------------------------------------------
#
# Notice that we did NOT choose one threshold for the comparison.
#
# For each classifier:
#
#     1. The classifier produces a score for every example.
#     2. We scan through possible thresholds.
#     3. Each threshold produces a different pair of metrics.
#     4. Those pairs become points on the ROC and PR curves.
#
# Because the two classifiers produce different scores, their
# threshold values are also different. Therefore, comparing
# the classifiers by their ROC/PR curves is more useful than
# comparing them at just one arbitrary threshold.
#
