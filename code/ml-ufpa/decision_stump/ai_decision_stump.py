"""Decision-stump example using Length and Weight train/test datasets."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DecisionStump:
    """A binary classifier that tests one feature against one threshold."""

    feature_index: int
    threshold: float
    label_at_or_below_threshold: int
    label_above_threshold: int

    def predict(self, features: npt.ArrayLike) -> npt.NDArray[np.int_]:
        """Classify rows of a feature matrix."""
        feature_array = np.asarray(features, dtype=float)
        selected_feature = feature_array[:, self.feature_index]
        return np.where(
            selected_feature <= self.threshold,
            self.label_at_or_below_threshold,
            self.label_above_threshold,
        )


def majority_label(labels: npt.NDArray[np.int_]) -> int:
    """Return the most common binary label; choose 0 when there is a tie."""
    return int(np.sum(labels) > len(labels) / 2)


def fit_decision_stump(
    features: npt.ArrayLike, labels: npt.ArrayLike
) -> tuple[DecisionStump, float]:
    """Train the best one-feature, one-threshold binary classifier."""
    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels, dtype=int).reshape(-1)

    if feature_array.ndim != 2 or len(feature_array) != len(label_array):
        raise ValueError("features must be a 2-D array with one row per label")
    if len(feature_array) == 0 or not np.all(np.isin(label_array, [0, 1])):
        raise ValueError("labels must be non-empty and contain only 0 and 1")

    best_stump = None
    best_error = float("inf")
    for feature_index in range(feature_array.shape[1]):
        values = feature_array[:, feature_index]
        unique_values = np.sort(np.unique(values))
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        for threshold in thresholds:
            left_mask = values <= threshold
            stump = DecisionStump(
                feature_index=feature_index,
                threshold=float(threshold),
                label_at_or_below_threshold=majority_label(label_array[left_mask]),
                label_above_threshold=majority_label(label_array[~left_mask]),
            )
            error = float(np.mean(stump.predict(feature_array) != label_array))
            if error < best_error:
                best_stump = stump
                best_error = error

    if best_stump is None:
        raise ValueError("at least one feature must have two distinct values")
    return best_stump, best_error


def accuracy(stump: DecisionStump, features: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Return the proportion of correctly classified samples."""
    return float(np.mean(stump.predict(features) == np.asarray(labels, dtype=int)))


def plot_decision_stump(
    train_features: npt.ArrayLike,
    train_labels: npt.ArrayLike,
    test_features: npt.ArrayLike,
    test_labels: npt.ArrayLike,
    stump: DecisionStump,
    feature_names: tuple[str, str],
) -> None:
    """Show train/test samples and the threshold learned from training data."""
    train_features_array = np.asarray(train_features, dtype=float)
    test_features_array = np.asarray(test_features, dtype=float)
    train_labels_array = np.asarray(train_labels, dtype=int)
    test_labels_array = np.asarray(test_labels, dtype=int)

    for label, color in ((0, "tab:blue"), (1, "tab:orange")):
        train_mask = train_labels_array == label
        test_mask = test_labels_array == label
        plt.scatter(
            train_features_array[train_mask, 0],
            train_features_array[train_mask, 1],
            color=color,
            edgecolors="black",
            label=f"train class {label}",
            s=80,
        )
        plt.scatter(
            test_features_array[test_mask, 0],
            test_features_array[test_mask, 1],
            color=color,
            marker="s",
            edgecolors="black",
            label=f"test class {label}",
            s=80,
        )

    if stump.feature_index == 0:
        plt.axvline(stump.threshold, color="black", linestyle="--", label="threshold")
    else:
        plt.axhline(stump.threshold, color="black", linestyle="--", label="threshold")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Decision stump trained on the training set")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def main() -> None:
    feature_names = ("Length", "Weight")
    train_features = np.array(
        [[12, 3.2], [10, 0.5], [14, 2.8], [14, 2.4], [13, 1.8], [13.8, 1.5], [11, 1]]
    )
    train_labels = np.array([0, 1, 0, 0, 1, 0, 1])
    test_features = np.array([[13, 3.1], [9, 0.8], [12.3, 1.4], [10, 2.3]])
    test_labels = np.array([0, 0, 1, 1])

    stump, train_error = fit_decision_stump(train_features, train_labels)
    print(f"Selected feature: {feature_names[stump.feature_index]}")
    print(f"Best threshold: {stump.threshold:.2f}")
    print(
        f"Rule: if {feature_names[stump.feature_index]} <= {stump.threshold:.2f}, "
        f"predict {stump.label_at_or_below_threshold}; otherwise, "
        f"predict {stump.label_above_threshold}."
    )
    print("Train predictions:", stump.predict(train_features))
    print(f"Train accuracy: {1 - train_error:.1%}")
    print("Test predictions: ", stump.predict(test_features))
    print(f"Test accuracy:  {accuracy(stump, test_features, test_labels):.1%}")

    plot_decision_stump(
        train_features,
        train_labels,
        test_features,
        test_labels,
        stump,
        feature_names,
    )


if __name__ == "__main__":
    main()
