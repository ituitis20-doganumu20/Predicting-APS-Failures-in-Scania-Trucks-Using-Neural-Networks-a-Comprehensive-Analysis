import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Load the CSV file into a pandas DataFrame
file_path = "aps_failure_training_set.csv"
df = pd.read_csv(file_path)

# Separate the target variable ("class") and the features
y = df["class"]
X = df.drop("class", axis=1)

# Convert all columns to numeric, replacing non-numeric values with NaN
X_numeric = X.apply(pd.to_numeric, errors='coerce')

mean_values = X_numeric.mean()
std_values = X_numeric.std()

# Define the threshold for identifying outliers
threshold = 2 * std_values

# Identify outliers
outliers = (X_numeric > mean_values + threshold) | (X_numeric < mean_values - threshold)

# Identify outliers in the modified X_numeric DataFrame
outliers_after_replacement = (X_numeric > mean_values + 2 * std_values) | (X_numeric < mean_values - 2 * std_values)

# Replace NaN values with the mean of each column
X_numeric_filled = X_numeric.fillna(X_numeric.mean())

# Identify features where min equals max
constant_features_mask = X_numeric_filled.min() == X_numeric_filled.max()
constant_features = X_numeric_filled.columns[constant_features_mask]

# Remove constant features from the DataFrame
X_numeric_filled.drop(columns=constant_features, inplace=True)

# Calculate the minimum and maximum values for each column
min_values = X_numeric_filled.min()
max_values = X_numeric_filled.max()

# Normalize the data to the [0, 1] range
X_numeric_normalized = (X_numeric_filled - X_numeric_filled.min()) / (X_numeric_filled.max() - X_numeric_filled.min())

# drop id since its not a feature
X_numeric_normalized.drop("id", axis=1, inplace=True)

# apply PCA

# Step 1: Calculate the covariance matrix
cov_matrix = np.cov(X_numeric_normalized, rowvar=False)

# Step 2: Calculate the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 3: Sort the eigenvectors based on the descending order of eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 4: Calculate the cumulative explained variance
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Step 5: Find the dimension where explained variance is 90%
target_explained_variance = 0.90
selected_dimensions = np.argmax(cumulative_explained_variance >= target_explained_variance) + 1

# Step 6: Choose the top 27 eigenvectors
top_eigenvectors = eigenvectors[:, :selected_dimensions]

# Step 7: Project the data onto the selected eigenvectors
X_pca = np.dot(X_numeric_normalized, top_eigenvectors)

# apply LDA after PCA 

# Map 'pos' to 1 and 'neg' to 0
y_binary = df['class'].map({'neg': 0, 'pos': 1})

S_W = np.sum([np.cov(X_pca[y_binary == label], rowvar=False) for label in np.unique(y_binary)], axis=0)

mean_class_1 = X_pca[y_binary == 0].mean(axis=0)
mean_class_2 = X_pca[y_binary == 1].mean(axis=0)
mean_class_1 = mean_class_1.reshape(-1, 1)
mean_class_2 = mean_class_2.reshape(-1, 1)

S_B = ((mean_class_1-mean_class_2) @ (mean_class_1-mean_class_2).T)

S_W_inv = np.linalg.inv(S_W)

multiplied_matrix= S_W_inv @ S_B

eigenvalues, eigenvectors = np.linalg.eig(multiplied_matrix)

# Sort the eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# Select the top 26 eigenvectors
transformation_matrix = eigenvectors_sorted[:, :26]
real_transformation_matrix=np.real(transformation_matrix) 

X_lda = X_pca @ real_transformation_matrix

# make predictions with neural network

X_lda = np.array(X_lda)
y_binary = np.array(y_binary)

# Define the neural network classifier 
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)

# Use Stratified K-Fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation with a progress bar
accuracy_scores = []

for train_index, test_index in tqdm(stratified_kfold.split(X_lda, y_binary), total=stratified_kfold.get_n_splits(), desc="Cross-Validation"):
    X_train, X_test = X_lda[train_index], X_lda[test_index]
    y_train, y_test = y_binary[train_index], y_binary[test_index]

    # Fit the model
    mlp_classifier.fit(X_train, y_train)

    # Evaluate the model on the test set
    accuracy = mlp_classifier.score(X_test, y_test)
    accuracy_scores.append(accuracy)

# Print the accuracy scores for each fold
print("Cross-Validation Accuracy Scores:", accuracy_scores)

# Print the mean and standard deviation of accuracy scores
print("Mean Accuracy:", np.mean(accuracy_scores))
print("Standard Deviation of Accuracy:", np.std(accuracy_scores))

""" Better score was gotten when commenting this part
# Fit the classifier on the entire dataset for detailed analysis


# Set the number of epochs for training
num_epochs = mlp_classifier.max_iter

# Create a progress bar for fitting
progress_bar = tqdm(total=num_epochs, desc="Training Progress", position=0, leave=True)

# Fit the classifier on the entire dataset with a progress bar
for epoch in range(num_epochs):
    # Fit the model for one epoch
    mlp_classifier.partial_fit(X_lda, y_binary, classes=np.unique(y_binary))

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# Get predictions on the entire dataset
y_pred = mlp_classifier.predict(X_lda)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_binary, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Generate classification report
class_report = classification_report(y_binary, y_pred)
print("Classification Report:\n", class_report)
"""

# make predictions on test data


# preprocess data
test_data = pd.read_csv('aps_failure_test_set.csv')

test_data_num= test_data.apply(pd.to_numeric, errors='coerce')
test_data_num.fillna(test_data_num.mean(), inplace=True)

# Find the dropped columns
dropped_columns = np.setdiff1d(test_data_num.columns, X_numeric_normalized.columns)

# Drop the corresponding columns from test_data_num
test_data_num = test_data_num.drop(columns=dropped_columns, axis=1)

test_pca = np.dot(test_data_num, top_eigenvectors)

test_lda= test_pca @ real_transformation_matrix

# make predictions and save csv file

predictions = mlp_classifier.predict(test_lda)

predicted_labels = ['neg' if label == 0 else 'pos' for label in predictions]

# Create a DataFrame for submission
submission_df = pd.DataFrame({'id': range(1, len(predictions) + 1), 'Predicted': predicted_labels})

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)



