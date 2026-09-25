import numpy as np

# 2 customers, each with 3 features: [Tenure, MonthlyCharges, ContractType]
X = np.array([
    [12, 50, 0],  # Customer 1
    [48, 90, 1]   # Customer 2
])

# Weights for 4 neurons in hidden layer (3 features -> 4 neurons)
np.random.seed(1)
W = np.random.randn(3, 4).round(2)
# print("\n",W)

# Compute linear pass
Z = np.dot(X, W)
# print("\n",Z)
print("Input Shape (X):", X.shape)
print("Weight Shape (W):", W.shape)
print("Output Shape (Z):", Z.shape)
print("\nCalculated Output (Z):\n", Z.round(2))


