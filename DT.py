import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import pickle
from sklearn.model_selection import KFold

df = pd.read_csv("cleaned_data_combined_modified.csv")[:-1]  # we drop the last empty row for training data

df_final = parse.parse_main(df)


# def print_problematic_rows(df):
#     """
#     Identify rows with NaN, infinity, or values exceeding float32's max.
#     Prints all such rows and returns them as a DataFrame.
#     """
#     # Check for NaN in any column
#     nan_mask = df.isna().any(axis=1)
#
#     # Check for infinity (-∞ or ∞) in any column (including object-type columns)
#     inf_mask = df.isin([np.inf, -np.inf]).any(axis=1)
#
#     # Check for values exceeding float32's maximum representable value in numeric columns
#     max_float32 = np.finfo('float32').max
#     numeric_cols = df.select_dtypes(include=np.number)
#     too_large_mask = (np.abs(numeric_cols) > max_float32).any(axis=1)
#
#     # Combine masks to find all problematic rows
#     combined_mask = nan_mask | inf_mask | too_large_mask
#     problematic_rows = df[combined_mask]
#
#     if not problematic_rows.empty:
#         print("Rows with NaN, Inf, or values too large for float32:")
#         print(problematic_rows)
#     else:
#         print("No problematic rows found.")
#
#     return problematic_rows
#
# print_problematic_rows(df_final)
# quit()

def save_sklearn_dt(model, filename):
    """Save sklearn Decision Tree to pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump({
            'tree': model.tree_,
            'classes': model.classes_
        }, f)


def load_dt_predictor(filename):
    """Load DT structure and create predictor function"""
    with open(filename, 'rb') as f:
        dt_data = pickle.load(f)

    tree = dt_data['tree']
    classes = dt_data['classes']

    # Extract tree structure
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold
    values = tree.value

    def predict(X):
        def predict_single(x):
            node = 0
            while children_left[node] != -1:  # While not leaf
                if x[features[node]] <= thresholds[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
            return classes[np.argmax(values[node])]

        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([predict_single(x) for x in X])

    return predict


def evaluate_classification(true_labels, pred_labels):
    # Calculate accuracy
    accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))

    # Create confusion matrix dataframe
    cm = pd.crosstab(pd.Series(true_labels, name='Actual'),
                     pd.Series(pred_labels, name='Predicted'),
                     margins=True)

    # Find most misclassified labels
    error_analysis = cm.drop('All', axis=1).drop('All', axis=0)
    np.fill_diagonal(error_analysis.values, 0)
    most_errors = error_analysis.sum(axis=1).sort_values(ascending=False)

    return accuracy, cm, most_errors


def plot_confusion_matrix(cm_df):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm_df.values[:-1, :-1], cmap='Blues')

    # Add labels
    classes = cm_df.index[:-1]
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Add values to cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm_df.iloc[i, j],
                     ha='center', va='center',
                     color='white' if cm_df.iloc[i, j] > cm_df.values.max() / 2 else 'black')

    plt.colorbar()
    plt.show()


def cross_val_accuracy(model, X, y, n_splits=5):
    """
    Perform k-fold cross validation and return validation accuracies
    Returns: (mean_accuracy, list of fold_accuracies)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        fold_acc = np.mean(preds == y_val)
        fold_accuracies.append(fold_acc)

    return np.mean(fold_accuracies), fold_accuracies


# train and load
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

df_target = df_final["target"]
df_feature = df_final.drop(columns="target", axis = 1)

# Train and save model
clf = DecisionTreeClassifier(max_depth=10).fit(df_feature, df_target)
save_sklearn_dt(clf, 'dt_model.pkl')


# Load predictor and make predictions
dt_predict = load_dt_predictor('dt_model.pkl')
X_new = df_feature.iloc[:10]
print("Predictions:", dt_predict(X_new))
print("Sklearn predictions:", clf.predict(X_new))

accuracy, cm, error_analysis = evaluate_classification(dt_predict(df_feature), df_target)

print(f"Accuracy: {accuracy:.2%}")
print("\nConfusion Matrix:")
print(cm)
print("\nMost Misclassified Labels:")
print(error_analysis)

# 5 fold CV
mean_acc, fold_accs = cross_val_accuracy(clf, df_feature, df_target, n_splits=5)

print("Fold Accuracies:", [f"{acc:.2%}" for acc in fold_accs])
print(f"Mean Validation Accuracy: {mean_acc:.2%}")
print(f"Std of Validation Accuracies: {np.std(fold_accs):.4f}")

# Plot
plot_confusion_matrix(cm)

# # crosstab to show drink vs target
# pd.crosstab(df_cleaned_Q6["Q6"], df_cleaned_Q6["target"]).plot(kind='bar', stacked=True)
# plt.title('Category vs Status Distribution')
# plt.tight_layout()
# plt.xlabel('Category')
# plt.ylabel('Count')
# plt.show()

