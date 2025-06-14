import numpy as np

def compute_bleu(reference, hypothesis, n_gram=4):
    def flatten(ref):
        if isinstance(ref, (list, tuple)) and len(ref) > 0 and isinstance(ref[0], (list, tuple)):
            return [item for sublist in ref for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        return ref
    reference = [flatten(r) for r in reference]
    precisions = []
    for n in range(1, n_gram+1):
        hyp_ngrams = [tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)]
        ref_ngrams = set(tuple(ref[i:i+n]) for ref in reference for i in range(len(ref)-n+1))
        match = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
        total = max(len(hyp_ngrams), 1)
        precisions.append(match / total)
    ref_lens = [len(ref) for ref in reference]
    hyp_len = len(hypothesis)
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
    bp = 1.0 if hyp_len > closest_ref_len else np.exp(1 - closest_ref_len / (hyp_len + 1e-8))
    bleu = bp * np.exp(np.sum([np.log(p+1e-8) for p in precisions]) / n_gram)
    return bleu

def normalize(X):
    return (X - np.mean(X)) / (np.std(X) + 1e-8)

def train_test_split(X, y, test_size=0.2):
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]
