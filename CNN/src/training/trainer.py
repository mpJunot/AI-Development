import numpy as np
from utils.early_stopping import EarlyStopping
from optimizers.sgd import SGD
from optimizers.adam import Adam
from utils.model_saver import ModelSaver

class Trainer:
    def __init__(self, model, optimizer_type="adam", learning_rate=0.01, batch_size=32, epochs=100,
                 early_stopping_patience=10, regularization_type=None, lambda_l1=0.0, lambda_l2=0.0,
                 save_dir="saved_models"):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.regularization_type = regularization_type
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.model_saver = ModelSaver(save_dir=save_dir)
        self.best_loss = float('inf')
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        if optimizer_type == "sgd":
            self.optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_type == "adam":
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None, save_best=True, checkpoint_freq=5):
        """
        Train the model with validation and history tracking.
        """
        num_samples = X_train.shape[0]
        self.best_loss = float('inf')

        for epoch in range(self.epochs):
            train_loss, train_accuracy = self._train_epoch(X_train, y_train)

            val_loss, val_accuracy = 0.0, 0.0
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self._validate(X_val, y_val)

            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            if X_val is not None:
                print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

            if save_best and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.model_saver.save_model(
                    model=self.model,
                    filename="best_model.pkl",
                    metadata={
                        'epoch': epoch,
                        'loss': val_loss,
                        'accuracy': val_accuracy
                    }
                )

            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.should_stop:
                print("Early stopping triggered.")
                break

        return self.history

    def _train_epoch(self, X, y):
        """Train for one epoch"""
        num_samples = X.shape[0]
        epoch_loss = 0
        epoch_accuracy = 0

        # Shuffle data
        indices = np.random.permutation(num_samples)
        X = X[indices]
        y = y[indices]

        for i in range(0, num_samples, self.batch_size):
            X_batch = X[i:i+self.batch_size]
            y_batch = y[i:i+self.batch_size]

            y_pred = self.model.forward(X_batch, training=True)

            loss = self.model.loss(y_pred, y_batch)
            reg_loss = self.model.compute_regularization_loss(
                self.regularization_type, self.lambda_l1, self.lambda_l2
            )
            total_loss = loss + reg_loss

            # Compute accuracy
            accuracy = self.model.accuracy(y_pred, y_batch)

            # Backward pass
            grad = y_pred
            grad[range(len(y_batch)), y_batch] -= 1
            grad /= len(y_batch)
            self.model.backward(grad)

            # Update parameters
            self.model.update(self.optimizer)

            epoch_loss += total_loss * len(y_batch)
            epoch_accuracy += accuracy * len(y_batch)

        return epoch_loss / num_samples, epoch_accuracy / num_samples

    def _validate(self, X, y):
        """Validate the model"""
        num_samples = X.shape[0]
        val_loss = 0
        val_accuracy = 0

        for i in range(0, num_samples, self.batch_size):
            X_batch = X[i:i+self.batch_size]
            y_batch = y[i:i+self.batch_size]

            y_pred = self.model.forward(X_batch, training=False)

            loss = self.model.loss(y_pred, y_batch)
            reg_loss = self.model.compute_regularization_loss(
                self.regularization_type, self.lambda_l1, self.lambda_l2
            )
            total_loss = loss + reg_loss

            accuracy = self.model.accuracy(y_pred, y_batch)

            val_loss += total_loss * len(y_batch)
            val_accuracy += accuracy * len(y_batch)

        return val_loss / num_samples, val_accuracy / num_samples

    def load_best_model(self):
        """Load the best model saved during training."""
        return self.model_saver.load_model("best_model.pkl", type(self.model))

    def load_checkpoint(self, checkpoint_name):
        """Load a specific checkpoint."""
        return self.model_saver.load_checkpoint(
            checkpoint_name,
            type(self.model),
            type(self.optimizer)
        )
