import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import pickle

df = pd.read_csv("cleaned_data_combined_modified.csv")[:-1]  # we drop the last empty row for training data

df_final = parse.parse_main(df)


from sklearn.neighbors import KNeighborsClassifier


#make the knn model on df_final
X = df_final.drop(columns=["target"],axis=1)
y = df_final["target"]

#normalize
mean = X.mean()
std = X.std()
X_norm = (X - mean) / std

knn = KNeighborsClassifier(n_neighbors=3).fit(X_norm, y)

#save the model as pkl file
filename = 'knn_model.pkl'
pickle.dump(knn, open(filename, 'wb'))



# Shuffle dataset
np.random.seed(42)
indices = np.random.permutation(len(X))

# Split indices into 5 equal folds
fold_size = len(X) // 5
fold_indices = [indices[i * fold_size:(i + 1) * fold_size] for i in range(5)]

# Store results
fold_accuracies = []
confusion_matrices = []

# Perform 5-fold cross-validation
for i in range(5):
    # Create train/test split manually
    test_idx = fold_indices[i]
    train_idx = np.hstack([fold_indices[j] for j in range(5) if j != i])
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Normalize using only the training data
    mean = X_train.mean()
    std = X_train.std()
    std[std == 0] = 1  # Avoid division by zero

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std  # Apply same scaling to test set

    # Train k-NN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_norm, y_train)

    # Predict on test set
    y_pred = knn.predict(X_test_norm)

    acc = np.mean(y_pred == y_test)
    fold_accuracies.append(acc)

    print(f"Fold {i+1} Accuracy: {acc:.2%}")

    #plot the confusion matrix without using functions
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']  )
    print(cm)



# Print overall results
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"Mean Validation Accuracy: {mean_acc:.2%}")
print(f"Std of Validation Accuracies: {std_acc:.4f}")






