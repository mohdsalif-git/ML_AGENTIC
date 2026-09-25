import numpy as np
import pandas as pd

# =====================================================================
# STEP 1: Generate & Preprocess Data using Pandas
# =====================================================================
np.random.seed(42)  # For reproducible results

n_samples = 1000
data = {
    'Tenure': np.random.randint(1, 72, size=n_samples),               # 1 to 72 months
    'MonthlyCharges': np.random.uniform(20.0, 120.0, size=n_samples), # $20 to $120
    'ContractType': np.random.choice([0, 1], size=n_samples)          # 0: Month-to-Month, 1: Annual
}

df = pd.DataFrame(data)

# Create ground truth churn targets based on realistic rule
churn_prob = 1 / (1 + np.exp(-(df['MonthlyCharges'] / 50 - df['Tenure'] / 10 - df['ContractType'] * 1.5)))
df['Churn'] = (churn_prob > 0.5).astype(int)

# Extract Features (X) and Labels (Y)
X_raw = df[['Tenure', 'MonthlyCharges', 'ContractType']].values
Y = df['Churn'].values.reshape(-1, 1)

# Z-Score Normalization: (X - mean) / std
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std

# Train/Test Split (80% Train, 20% Test)
split = int(n_samples * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print(f"Data Prep Complete | Train Set: {X_train.shape} | Test Set: {X_test.shape}\n")


# =====================================================================
# STEP 2: Activations & Derivatives
# =====================================================================
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -500, 500)  # Numerical overflow protection
    return 1 / (1 + np.exp(-z))


# =====================================================================
# STEP 3: Parameter Initialization
# =====================================================================
input_dim = 3    # Tenure, MonthlyCharges, ContractType
hidden_dim = 4   # Hidden layer neurons
output_dim = 1   # Binary probability output

# He Initialization for ReLU layer
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros((1, hidden_dim))

# Xavier Initialization for Sigmoid layer
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
b2 = np.zeros((1, output_dim))

learning_rate = 0.05
epochs = 2000
m = len(X_train)


# =====================================================================
# STEP 4: Training Loop
# =====================================================================
for epoch in range(1, epochs + 1):
    
    # Forward Pass
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # Compute Binary Cross-Entropy Loss
    epsilon = 1e-15
    loss = -np.mean(Y_train * np.log(A2 + epsilon) + (1 - Y_train) * np.log(1 - A2 + epsilon))
    
    # Backpropagation
    dZ2 = A2 - Y_train
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X_train.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    # Gradient Descent Updates
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    if epoch % 400 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")


# =====================================================================
# STEP 5: Model Evaluation on Test Data
# =====================================================================
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
test_probs = sigmoid(Z2_test)

predictions = (test_probs >= 0.5).astype(int)
accuracy = np.mean(predictions == Y_test) * 100

print(f"\n==========================================")
print(f"FINAL TEST ACCURACY: {accuracy:.2f}%")
print(f"==========================================\n")

# Display first 5 customer predictions
results = pd.DataFrame({
    'Actual Churn': Y_test.flatten()[:5],
    'Predicted Prob': test_probs.flatten()[:5].round(3),
    'Predicted Class': predictions.flatten()[:5]
})
print("Sample Model Outputs:")
print(results)