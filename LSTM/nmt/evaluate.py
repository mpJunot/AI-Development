import numpy as np
from core.utils import compute_bleu

def evaluate_bleu(X_src, X_tgt, model):
    y_pred = model.forward(X_src, X_tgt)
    bleu = 0
    batch_size = X_tgt.shape[1]
    for i in range(batch_size):
        ref = [X_tgt[:, i, 0].tolist()]
        hyp = np.argmax(y_pred[:, i, :], axis=1).tolist()
        bleu += compute_bleu(ref, hyp)
    bleu /= batch_size
    return bleu
