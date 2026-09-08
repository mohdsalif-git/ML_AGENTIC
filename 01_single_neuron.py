import numpy as np

def sigmoid(z):
    """Activation function: Squeezes any real number into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-z))

class SingleNeuron:
    def __init__(self, input_size):
        # 1. Initialize random weights and zero bias
        self.weights = np.random.randn(input_size)
        self.bias = 0.0

    def forward(self, inputs):
        # 2. Weighted sum: (x1*w1 + x2*w2 + ...) + bias
        z = np.dot(inputs, self.weights) + self.bias
        # 3. Apply non-linear activation
        return sigmoid(z)

# --- Execute Test Run ---
if __name__ == "__main__":
    # Create a neuron expecting 3 input features
    neuron = SingleNeuron(input_size=3)
    
    # Input vector (e.g., 3 numeric features)
    sample_input = np.array([0.5, 1.2, -0.8])
    
    output = neuron.forward(sample_input)
    
    print(f"Weights: {neuron.weights}")
    print(f"Bias:    {neuron.bias}")
    print(f"Output Prediction: {output:.4f}")