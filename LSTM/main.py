import numpy as np
import matplotlib.pyplot as plt
import os
from datasets import load_dataset
from transformers import AutoTokenizer

from core.lstm import LSTM
from core.optimizers import SGD
from core.trainer import Trainer
from nmt.evaluate import evaluate_bleu

dataset = load_dataset("wmt14", "de-en")
train_data = dataset['train'].select(range(10000))
val_data = dataset['validation'].select(range(1000))

tokenizer_src = AutoTokenizer.from_pretrained("t5-small")
tokenizer_tgt = AutoTokenizer.from_pretrained("t5-small")
max_length = 32

def encode_pair(example):
    src = tokenizer_src(
        example['translation']['en'],
        truncation=True, padding='max_length', max_length=max_length, return_tensors='np'
    )
    tgt = tokenizer_tgt(
        example['translation']['de'],
        truncation=True, padding='max_length', max_length=max_length, return_tensors='np'
    )
    return {
        'src_input_ids': src['input_ids'][0],
        'tgt_input_ids': tgt['input_ids'][0]
    }

train_data = train_data.map(encode_pair)
val_data = val_data.map(encode_pair)

def get_batch(data, batch_size=16):
    for i in range(0, len(data), batch_size):
        batch = data.select(range(i, min(i+batch_size, len(data))))
        X_src = np.stack([ex['src_input_ids'] for ex in batch], axis=1)
        X_tgt = np.stack([ex['tgt_input_ids'] for ex in batch], axis=1)
        X_src = X_src[:, :, np.newaxis]
        X_tgt = X_tgt[:, :, np.newaxis]
        yield X_src, X_tgt

class SimpleSeq2Seq:
    def __init__(self, input_size, hidden_size, output_size):
        self.encoder = LSTM(input_size, hidden_size)
        self.decoder = LSTM(input_size, hidden_size)
        self.W = np.random.randn(output_size, hidden_size) * 0.01
        self.b = np.zeros((output_size, 1))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.hidden_size = hidden_size
        self.last_input = None
        self.last_encoder_input_shape = None

    def forward(self, X_src, X_tgt):
        batch_size = X_src.shape[1]
        h0 = np.zeros((batch_size, self.hidden_size))
        c0 = np.zeros((batch_size, self.hidden_size))
        encoder_outputs, _ = self.encoder.forward(X_src, h0, c0)
        seq_len, batch, _ = X_tgt.shape
        h_prev = encoder_outputs[-1]
        c_prev = np.zeros((batch, self.hidden_size))
        outputs = []
        for t in range(seq_len):
            h_dec, c_dec = self.decoder.forward(X_tgt[t:t+1], h_prev, c_prev)
            h_t = h_dec[-1]
            self.last_input = h_t
            out = self.W @ h_t.T + self.b
            outputs.append(out.T)
            h_prev, c_prev = h_t, c_dec[-1]
        outputs = np.stack(outputs, axis=0)
        self.last_encoder_input_shape = X_src.shape
        return outputs

    def backward(self, dy):
        seq_len, batch_size, _ = dy.shape
        dh_dec = np.zeros((seq_len, batch_size, self.hidden_size))
        for t in range(seq_len):
            dy_t = dy[t].T
            dh = self.W.T @ dy_t
            dh_dec[t] = dh.T
            self.dW += dy_t @ self.last_input / batch_size
            self.db += np.sum(dy_t, axis=1, keepdims=True) / batch_size
        ddecoder_out = self.decoder.backward(dh_dec, np.zeros_like(dh_dec))
        if isinstance(ddecoder_out, (tuple, list)):
            ddecoder = ddecoder_out[0]
        else:
            ddecoder = ddecoder_out
        seq_len_src = self.last_encoder_input_shape[0]
        batch = ddecoder.shape[1]
        hidden = ddecoder.shape[2]
        dh_enc = np.zeros((seq_len_src, batch, hidden))
        dh_enc[-1] = ddecoder[-1]
        dc_enc = np.zeros_like(dh_enc)
        dX_src = self.encoder.backward(dh_enc, dc_enc)
        if isinstance(dX_src, (tuple, list)):
            dX_src = dX_src[0]
        return dX_src

    def get_params_and_grads(self):
        params = {}
        for name, (p, g) in self.encoder.get_params_and_grads().items():
            params[f'encoder_{name}'] = (p, g)
        for name, (p, g) in self.decoder.get_params_and_grads().items():
            params[f'decoder_{name}'] = (p, g)
        params['W'] = (self.W, self.dW)
        params['b'] = (self.b, self.db)
        return params

    def save_weights(self, filename):
        np.savez_compressed(
            filename,
            encoder_W=self.encoder.W,
            encoder_U=self.encoder.U,
            encoder_b=self.encoder.b,
            decoder_W=self.decoder.W,
            decoder_U=self.decoder.U,
            decoder_b=self.decoder.b,
            W=self.W,
            b=self.b
        )

    def load_weights(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Le fichier {filename} n'existe pas")
        data = np.load(filename)
        self.encoder.W = data['encoder_W']
        self.encoder.U = data['encoder_U']
        self.encoder.b = data['encoder_b']
        self.decoder.W = data['decoder_W']
        self.decoder.U = data['decoder_U']
        self.decoder.b = data['decoder_b']
        self.W = data['W']
        self.b = data['b']

    def predict(self, X_src, max_len=32, sos_token_id=0, eos_token_id=1):
        batch_size = X_src.shape[1]
        h0 = np.zeros((batch_size, self.hidden_size))
        c0 = np.zeros((batch_size, self.hidden_size))
        encoder_outputs, _ = self.encoder.forward(X_src, h0, c0)
        h_prev = encoder_outputs[-1]
        c_prev = np.zeros((batch_size, self.hidden_size))
        current_token = np.full((1, batch_size, 1), sos_token_id)
        outputs = []
        for _ in range(max_len):
            h_dec, c_dec = self.decoder.forward(current_token, h_prev, c_prev)
            h_t = h_dec[-1]
            out = self.W @ h_t.T + self.b
            out = out.T
            next_token = np.argmax(out, axis=-1)
            outputs.append(next_token)
            current_token = next_token[:, np.newaxis, np.newaxis]
            h_prev, c_prev = h_t, c_dec[-1]
            if np.all(next_token == eos_token_id):
                break
        outputs = np.stack(outputs, axis=0).T
        return outputs

def cross_entropy_loss(y_pred, y_true, pad_token_id=0):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    seq_len, batch = y_true.shape
    losses = []
    for t in range(seq_len):
        for b in range(batch):
            idx = y_true[t, b]
            if idx == pad_token_id:
                continue
            losses.append(-np.log(y_pred[t, b, idx]))
    return np.mean(losses) if losses else 0.0

input_size = 1
hidden_size = 128
output_size = tokenizer_tgt.vocab_size
model = SimpleSeq2Seq(input_size, hidden_size, output_size)
try:
    model.load_weights('weights.npz')
except FileNotFoundError:
    pass
optimizer = SGD(learning_rate=0.01, momentum=0.9)
loss_fn = cross_entropy_loss
trainer = Trainer(model, optimizer, loss_fn)

num_epochs = 10
batch_size = 16
losses = []
bleu_scores = []

for epoch in range(num_epochs):
    epoch_losses = []
    for X_src, X_tgt in get_batch(train_data, batch_size=batch_size):
        loss = trainer.fit((X_src, X_tgt), X_tgt, epochs=1, verbose=1)
        epoch_losses.append(loss)
    mean_loss = np.mean(epoch_losses)
    losses.append(mean_loss)

    val_bleu_scores = []
    for X_src, X_tgt in get_batch(val_data, batch_size=16):
        bleu = evaluate_bleu(X_src, X_tgt, model)
        val_bleu_scores.append(bleu)
    mean_bleu = np.mean(val_bleu_scores)
    bleu_scores.append(mean_bleu)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {mean_loss:.4f} - Validation BLEU: {mean_bleu:.4f}")

model.save_weights('weights.npz')

plt.figure()
plt.plot(losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('loss_curve.png')

plt.figure()
plt.plot(bleu_scores, label='Validation BLEU')
plt.xlabel('Epoch')
plt.ylabel('BLEU score')
plt.title('Validation BLEU Score Curve')
plt.legend()
plt.savefig('bleu_curve.png')
