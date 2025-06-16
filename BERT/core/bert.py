import numpy as np
from typing import List, Tuple, Dict, Optional

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = np.random.normal(0, 0.02, (d_model, d_model))
        self.W_k = np.random.normal(0, 0.02, (d_model, d_model))
        self.W_v = np.random.normal(0, 0.02, (d_model, d_model))
        self.W_o = np.random.normal(0, 0.02, (d_model, d_model))

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def combine_heads(self, x, batch_size):
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch_size, -1, self.d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = np.matmul(q, k.transpose(0, 1, 3, 2))
        dk = np.sqrt(self.head_dim)
        scaled_attention_logits = matmul_qk / dk

        if mask is not None:
            mask = np.expand_dims(mask, axis=1)
            mask = np.expand_dims(mask, axis=2)
            scaled_attention_logits += (mask * -1e9)

        attention_weights = self.softmax(scaled_attention_logits, axis=-1)
        output = np.matmul(attention_weights, v)
        return output, attention_weights

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def call(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = np.matmul(q, self.W_q)
        k = np.matmul(k, self.W_k)
        v = np.matmul(v, self.W_v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        concat_attention = self.combine_heads(scaled_attention, batch_size)
        output = np.matmul(concat_attention, self.W_o)

        return output, attention_weights

class FeedForward:
    def __init__(self, d_model, dff):
        self.dense1 = np.random.normal(0, 0.02, (d_model, dff))
        self.dense2 = np.random.normal(0, 0.02, (dff, d_model))

    def call(self, x):
        x = np.matmul(x, self.dense1)
        x = self.gelu(x)
        x = np.matmul(x, self.dense2)
        return x

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

class TransformerLayer:
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
    ):
        self.mha = MultiHeadAttention(hidden_size, num_attention_heads)
        self.ffn = FeedForward(hidden_size, intermediate_size)

        self.layernorm1 = LayerNormalization(hidden_size)
        self.layernorm2 = LayerNormalization(hidden_size)

        self.dropout1 = Dropout(hidden_dropout_prob)
        self.dropout2 = Dropout(hidden_dropout_prob)

    def forward(self, x, attention_mask=None):
        attn_output, _ = self.mha.call(x, x, x, attention_mask)
        attn_output = self.dropout1.call(attn_output, training=True)
        out1 = self.layernorm1.call(x + attn_output)

        ffn_output = self.ffn.call(out1)
        ffn_output = self.dropout2.call(ffn_output, training=True)
        out2 = self.layernorm2.call(out1 + ffn_output)

        return out2

class LayerNormalization:
    def __init__(self, d_model, epsilon=1e-6):
        self.epsilon = epsilon
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def call(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * x_norm + self.beta

class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def call(self, x, training):
        if training:
            mask = np.random.binomial(1, 1-self.rate, size=x.shape) / (1-self.rate)
            return x * mask
        return x

class BERT:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.token_embeddings = np.random.normal(0, 0.02, (vocab_size, hidden_size))
        self.position_embeddings = np.random.normal(0, 0.02, (max_position_embeddings, hidden_size))
        self.token_type_embeddings = np.random.normal(0, 0.02, (type_vocab_size, hidden_size))

        self.transformer_layers = []
        for _ in range(num_hidden_layers):
            self.transformer_layers.append(TransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob
            ))

        self.layer_norm = LayerNormalization(hidden_size)

    def forward(
        self,
        input_ids: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, sequence_length = input_ids.shape

        token_embeddings = self.token_embeddings[input_ids]
        position_embeddings = self.position_embeddings[:sequence_length]
        position_embeddings = np.expand_dims(position_embeddings, axis=0)
        position_embeddings = np.tile(position_embeddings, (batch_size, 1, 1))

        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings[token_type_ids]

        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm.call(embeddings)

        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer.forward(hidden_states, attention_mask)

        pooled_output = np.mean(hidden_states, axis=1)

        return hidden_states, pooled_output

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.embedding = np.random.normal(0, 0.02, (vocab_size, d_model))

    def call(self, x):
        return self.embedding[x]

class PositionalEncoding:
    def __init__(self, position, d_model):
        self.position = position
        self.d_model = d_model

    def call(self, seq_len):
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        pos_encoding = np.zeros((seq_len, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        return pos_encoding
