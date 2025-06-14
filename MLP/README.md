# Multi-Layer Perceptron (MLP) Implementation

This project implements a Multi-Layer Perceptron neural network from scratch using NumPy. The implementation includes various features for training and evaluating neural networks.

## Features

- Multiple activation functions:
  - ReLU
  - Sigmoid
  - Tanh
- L2 regularization for preventing overfitting
- Early stopping mechanism
- Model saving and loading capabilities
- Confusion matrix evaluation
- Training with validation data support
- Customizable learning rate
- Layer-wise architecture

## Requirements

- Python 3.x
- NumPy

## Installation
1. Setup Environment

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

```python
from Mlp import MLP

# Create an MLP with layer sizes [input_size, hidden_size, output_size]
mlp = MLP(layer_sizes=[784, 128, 10], activation_function="relu", learning_rate=0.1)

# Train the network
mlp.train(
    training_data=X_train,
    labels=y_train,
    epochs=100,
    validation_data=X_val,
    validation_labels=y_val,
    l2_lambda=0.01,
    patience=5
)

# Make predictions
predictions = mlp.predict(X_test)

# Evaluate using confusion matrix
confusion_matrix = mlp.confusion_matrix(X_test, y_test)

# Save the model
mlp.save_model("model.npy")
```

## Project Structure

- `code/`
  - `Mlp.py`: Main MLP implementation
  - `Layer.py`: Layer class implementation
- `docs/`: Documentation and assignment details

## License

This project is part of a machine learning assignment and is for educational purposes.
