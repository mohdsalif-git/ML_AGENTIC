import numpy as np

actual_y = np.array([1, 0, 1, 0])          # True labels
predicted_y = np.array([0.9, 0.1, 0.2, 0.8]) # Predicted probabilities

# Binary Cross-Entropy formula
epsilon = 1e-15  # Avoids log(0)
loss = -np.mean(actual_y * np.log(predicted_y + epsilon) + (1 - actual_y) * np.log(1 - predicted_y + epsilon))

print("Actual Targets:   ", actual_y)
print("Model Predictions:", predicted_y)
print(f"Loss Value:        {loss:.4f}")

