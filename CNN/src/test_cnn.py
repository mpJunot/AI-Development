import numpy as np
from model.lenet5 import LeNet5
from training.trainer import Trainer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.datasets import fetch_openml

def load_mnist():
    """
    Load the MNIST dataset using sklearn
    """
    print("Loading MNIST from sklearn...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    y = y.astype(np.int32)

    X = X.astype('float32') / 255.0

    X = X.reshape(-1, 1, 28, 28)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return (X_train, y_train), (X_test, y_test)

def plot_training_history(history):
    """Plot the training history"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def visualize_predictions(model, X_test, y_test, num_samples=5):
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_samples = X_test[indices]
    y_true = y_test[indices]

    y_pred = model.predict(X_samples)

    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_samples[i, 0], cmap='gray')
        plt.title(f'True: {y_true[i]}\nPred: {y_pred[i]}')
        plt.axis('off')
    plt.savefig('predictions.png')
    plt.close()

def main():
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_mnist()

    print("Initializing model...")
    model = LeNet5(input_shape=(1, 28, 28), num_classes=10)

    trainer = Trainer(
        model=model,
        optimizer_type="adam",
        learning_rate=0.001,
        batch_size=32,
        epochs=20,
        save_dir="saved_models"
    )

    print("Starting training...")
    history = trainer.train(
        X_train, y_train,
        X_val=X_test[:1000], y_val=y_test[:1000],
        save_best=True,
        checkpoint_freq=5
    )

    plot_training_history(history)

    print("\nModel evaluation...")
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy on test set: {accuracy:.4f}")

    plot_confusion_matrix(y_test, y_pred)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nVisualizing some predictions...")
    visualize_predictions(model, X_test, y_test)

    trainer.model_saver.save_model(
        model=model,
        filename="final_model.pkl",
        metadata={
            'accuracy': accuracy,
            'input_shape': model.input_shape,
            'num_classes': model.num_classes
        }
    )

    print("\nTest completed! Check the generated files:")
    print("- training_history.png: Training history")
    print("- confusion_matrix.png: Confusion matrix")
    print("- predictions.png: Visualization of some predictions")
    print("- saved_models/final_model.pkl: Model saved")

if __name__ == "__main__":
    main()
