from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import pickle
from sklearn.model_selection import KFold

df = pd.read_csv("cleaned_data_combined_modified.csv")[:-1]  # we drop the last empty row for training data

df_final = parse.parse_main(df)

df_target = df_final["target"]
df_feature = df_final.drop(columns="target", axis=1)

# Set random seed for reproducibility
np.random.seed(42)

# Define the parameter grid with max_depth None replaced by 100
param_grid = {
    'max_depth': [7, 10, 13, 16, 20, 23],
    'n_estimators': [80, 100, 120, 140, 160, 180, 200, 220]
}

# Create base Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Set up grid search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf,
                          param_grid=param_grid,
                          cv=5,
                          scoring='accuracy',
                          n_jobs=-1,
                          verbose=1)

# Perform grid search
print("Performing grid search...")
grid_search.fit(df_feature, df_target)

# Get results
results = grid_search.cv_results_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print best parameters
print(f"\nBest parameters: {best_params}")
print(f"Best validation accuracy: {best_score:.4f}")

# Prepare data for heatmap
max_depth_values = param_grid['max_depth']
n_estimators_values = param_grid['n_estimators']

# Reshape scores into matrix (max_depth x n_estimators)
mean_scores = results['mean_test_score'].reshape(
    len(max_depth_values),
    len(n_estimators_values)
)

# Create heatmap plot
plt.figure(figsize=(10, 6))
im = plt.imshow(mean_scores, cmap='viridis', aspect='auto')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Validation Accuracy', rotation=270, labelpad=15)

# Set ticks and labels
plt.xticks(np.arange(len(n_estimators_values)), labels=n_estimators_values)
plt.yticks(np.arange(len(max_depth_values)), labels=max_depth_values)

# Add labels and title
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Max Depth')
plt.title('Random Forest Hyperparameter Tuning Heatmap\n(5-Fold CV Accuracy)')

# Annotate heatmap with accuracy values
for i in range(len(max_depth_values)):
    for j in range(len(n_estimators_values)):
        text = plt.text(j, i, f'{mean_scores[i, j]:.3f}',
                       ha="center", va="center",
                       color="w" if mean_scores[i, j] < 0.8 else "k")

plt.tight_layout()
plt.savefig('rf_heatmap.png', dpi=300, bbox_inches='tight')
print("\nHeatmap saved as rf_heatmap.png")
plt.show()