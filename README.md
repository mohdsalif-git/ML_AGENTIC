# 🧠 Customer Churn Prediction Neural Network (Built from Scratch)

A 3-layer feedforward artificial neural network implementation built entirely from scratch using **Python**, **NumPy**, and **Pandas** without high-level deep learning frameworks like PyTorch or TensorFlow. 

This project demonstrates the mathematical fundamentals of deep learning—including forward propagation, loss computation, backpropagation, and gradient descent—applied to a binary classification problem: predicting telecom customer churn.

---

## 🛠️ Key Features & Architecture

* **Zero Frameworks:** Written using pure mathematical operations via NumPy matrix vectorization.
* **Architecture:** `3 Input Features` ➡️ `Hidden Layer (4 Neurons, ReLU)` ➡️ `Output Layer (1 Neuron, Sigmoid)`.
* **Robust Initialization:** Uses **He Initialization** for the ReLU layer and **Xavier Initialization** for the Sigmoid output layer.
* **Feature Normalization:** Applies Z-score normalization ($X = \frac{X - \mu}{\sigma}$) to prevent exploding/vanishing gradients.
* **Custom Functions:** Manual implementations of vectorized ReLU, Sigmoid, Binary Cross-Entropy Loss, and Backpropagation derivatives.

---

## 📊 Dataset Features

| Feature | Type | Description |
| :--- | :--- | :--- |
| `Tenure` | Continuous | Months the customer has stayed with the company ($1–72$) |
| `MonthlyCharges` | Continuous | Amount billed per month ($20–$120 USD) |
| `ContractType` | Categorical | $0$ = Month-to-Month, $1$ = One-Year Contract |
| **`Churn` (Target)** | Binary Class | $0$ = Retained, $1$ = Churned |

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with NumPy and Pandas:

```bash
pip install numpy pandas
