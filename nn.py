import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from clean_q2 import extract_number
from clean_Q4 import replace_price_Q4
from clean_Q6 import clean_Q6

# Load and clean data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Apply the cleaning function to the second column
    df.iloc[:, 2] = df.iloc[:, 2].apply(extract_number)
    
    # Compute the average of non-None values
    valid_values = [v for v in df.iloc[:, 2] if pd.notna(v)]
    average_value = sum(valid_values) // len(valid_values) if valid_values else 0
    
    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: x if pd.notna(x) else average_value)
    
    # Apply Q4 Cleaning
    if 'Q4: How much would you expect to pay for one serving of this food item?' in df.columns:
        df['Q4'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(replace_price_Q4)
        numeric_series = pd.to_numeric(df['Q4'], errors='coerce')
        df['Q4'] = df['Q4'].apply(lambda x: x if pd.notna(x) else numeric_series.mean())
    
    # Apply Q6 Cleaning
    df = clean_Q6(df)
    
    return df

# Neural Network class
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def backward(self, X, Y, output):
        m = X.shape[0]
        dZ2 = output - Y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.W2.T) * (self.A1 * (1 - self.A1))
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def compute_loss(self, Y, output):
        return -np.mean(np.sum(Y * np.log(output + 1e-9), axis=1))
    
    def train(self, X_train, Y_train, X_val, Y_val, epochs=1000):
        training_loss = []
        validation_accuracy = []
        for epoch in range(epochs):
            output = self.forward(X_train)
            self.backward(X_train, Y_train, output)
            loss = self.compute_loss(Y_train, output)
            training_loss.append(loss)
            
            if epoch % 100 == 0:
                val_pred = np.argmax(self.forward(X_val), axis=1)
                val_true = np.argmax(Y_val, axis=1)
                accuracy = np.mean(val_pred == val_true)
                validation_accuracy.append(accuracy)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        
        return training_loss, validation_accuracy
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Prepare data
def prepare_data(df, label_column):
    X = df.drop(columns=[label_column])
    y = df[label_column]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # One-hot encode labels
    y_train = np.eye(len(set(y)))[y_train]
    y_val = np.eye(len(set(y)))[y_val]
    y_test = np.eye(len(set(y)))[y_test]
    
    return X_train.to_numpy(), y_train, X_val.to_numpy(), y_val, X_test.to_numpy(), y_test, label_encoder

# Main function
def main():
    file_path = "cleaned_data_combined.csv"
    df = load_and_clean_data(file_path)
    
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = prepare_data(df, 'label')
    
    input_size = X_train.shape[1]
    hidden_size = 10
    output_size = y_train.shape[1]
    
    nn = SimpleNN(input_size, hidden_size, output_size, learning_rate=0.1)
    
    loss, val_acc = nn.train(X_train, y_train, X_val, y_val, epochs=1000)
    
    # Evaluate on test set
    y_pred = nn.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
if __name__ == "__main__":
    main()
