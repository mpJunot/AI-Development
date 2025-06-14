
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Early stopping to terminate training when validation loss stops improving.
        :param patience: Number of epochs to wait after the last improvement.
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_loss: float):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
