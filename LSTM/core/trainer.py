import numpy as np

class Trainer:
    def __init__(self, model, optimizer, loss_fn, dropout=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dropout = dropout

    def fit(self, X, y, epochs=10, batch_size=None, verbose=1):
        if isinstance(X, tuple):
            X_src, X_tgt = X
            n_samples = X_src.shape[1]
        else:
            X_src = X
            X_tgt = None
            n_samples = X_src.shape[1]
        for epoch in range(epochs):
            if self.dropout:
                X_src_batch = self.dropout.forward(X_src, training=True)
                self.dropout.reset()
            else:
                X_src_batch = X_src
            if X_tgt is not None:
                y_pred = self.model.forward(X_src_batch, X_tgt)
                dy = cross_entropy_backward(y_pred, y.squeeze(-1))
                self.model.backward(dy)
            else:
                h0 = np.zeros((n_samples, self.model.hidden_size))
                if hasattr(self.model, 'forward') and 'c0' in self.model.forward.__code__.co_varnames:
                    c0 = np.zeros((n_samples, self.model.hidden_size))
                    h, c = self.model.forward(X_src_batch, h0, c0)
                    y_pred = h[-1, :, 0] if h.shape[2] == 1 else h[-1]
                else:
                    h, h_last = self.model.forward(X_src_batch, h0)
                    y_pred = h[-1, :, 0] if h.shape[2] == 1 else h[-1]
                dy = cross_entropy_backward(y_pred, y.squeeze(-1))
                dh = np.zeros_like(h)
                dh[-1, :, 0] = dy if h.shape[2] == 1 else dy
                if hasattr(self.model, 'forward') and 'c0' in self.model.forward.__code__.co_varnames:
                    dc = np.zeros_like(c)
                    self.model.backward(dh, dc)
                else:
                    self.model.backward(dh)
            loss = self.loss_fn(y_pred, y.squeeze(-1))
            if verbose and (epoch+1) % max(1, epochs//10) == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
            params_grads = self.model.get_params_and_grads()
            self.optimizer.step(
                {k: v[0] for k, v in params_grads.items()},
                {k: v[1] for k, v in params_grads.items()}
            )
            return loss

    def evaluate(self, X, y):
        n_samples = X.shape[1]
        h0 = np.zeros((n_samples, self.model.hidden_size))
        if hasattr(self.model, 'forward') and 'c0' in self.model.forward.__code__.co_varnames:
            c0 = np.zeros((n_samples, self.model.hidden_size))
            h, c = self.model.forward(X, h0, c0)
            y_pred = h[-1, :, 0] if h.shape[2] == 1 else h[-1]
        else:
            h, h_last = self.model.forward(X, h0)
            y_pred = h[-1, :, 0] if h.shape[2] == 1 else h[-1]
        loss = self.loss_fn(y_pred, y.flatten())
        return loss, y_pred

def cross_entropy_backward(y_pred, y_true):
    grad = np.copy(y_pred)
    seq_len, batch = y_true.shape
    for t in range(seq_len):
        for b in range(batch):
            idx = y_true[t, b]
            grad[t, b, idx] -= 1
    grad /= (seq_len * batch)
    return grad
