# LeNet5 Implementation

This project is an implementation of the LeNet5 architecture for image classification, specifically designed for the MNIST dataset. The implementation is done in Python using only NumPy

## Project Structure

```
.
├── src/
│   ├── model/
│   │   └── lenet5.py
│   ├── training/
│   │   └── trainer.py
│   ├── optimizers/
│   │   ├── adam.py
│   │   ├── sgd.py
│   │   └── regularization.py
│   ├── utils/
│   │   ├── model_saver.py
│   │   ├── initializers.py
│   │   └── early_stopping.py
│   ├── main.py
│   └── test_cnn.py
├── saved_models/
├── requirements.txt
└── README.md
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train and test the model:

```bash
python src/test_cnn.py
```

The script will:
1. Load the MNIST dataset
2. Train the LeNet5 model
3. Generate training visualizations
4. Save the trained model

## Results

The script generates several files:
- `training_history.png`: Graph showing the evolution of loss and accuracy during training
- `confusion_matrix.png`: Confusion matrix on the test set
- `predictions.png`: Visualization of some predictions
- `saved_models/final_model.pkl`: Saved model

## LeNet5 Architecture

The LeNet5 architecture includes:
- 2 convolutional layers
- 2 pooling layers
- 3 fully connected layers
- ReLU activation function
- Softmax output

## Implementation Details

### Core Components

1. **Model Architecture (`model/lenet5.py`)**
   - Implements the complete LeNet5 architecture using NumPy
   - Supports both forward and backward propagation
   - Includes all necessary layers: Conv2D, MaxPooling2D, Dense
   - Uses ReLU activation and Softmax for classification

2. **Training System (`training/trainer.py`)**
   - Handles the complete training loop
   - Implements batch processing
   - Supports validation during training
   - Includes progress tracking and metrics logging

3. **Optimizers (`optimizers/`)**
   - `adam.py`: Implements Adam optimizer with momentum and adaptive learning rates
   - `sgd.py`: Implements Stochastic Gradient Descent
   - `regularization.py`: Provides L1 and L2 regularization options

4. **Utilities (`utils/`)**
   - `model_saver.py`: Handles model serialization and loading
   - `initializers.py`: Provides weight initialization methods
   - `early_stopping.py`: Implements early stopping to prevent overfitting

### Key Features

1. **Memory Efficiency**
   - Implements batch processing to handle large datasets
   - Uses NumPy's efficient array operations
   - Minimizes memory usage during training

2. **Training Features**
   - Supports multiple optimizers (Adam, SGD)
   - Implements learning rate scheduling
   - Includes regularization options
   - Provides early stopping mechanism

3. **Monitoring and Visualization**
   - Tracks training and validation metrics
   - Generates training history plots
   - Creates confusion matrix visualization
   - Shows sample predictions

4. **Model Management**
   - Saves and loads model checkpoints
   - Preserves training history
   - Supports model evaluation on test data

### Performance Considerations

1. **Computation**
   - Uses vectorized operations for efficiency
   - Implements efficient backpropagation
   - Optimizes memory usage during training

2. **Training**
   - Supports mini-batch processing
   - Implements efficient gradient computation
   - Uses optimized matrix operations

3. **Memory**
   - Minimizes memory footprint
   - Efficiently handles large datasets
   - Implements proper cleanup of temporary variables
