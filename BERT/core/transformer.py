import numpy as np
from typing import Optional, Tuple

class MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.query = np.random.normal(0, 0.02, (hidden_size, hidden_size))
        self.key = np.random.normal(0, 0.02, (hidden_size, hidden_size))
        self.value = np.random.normal(0, 0.02, (hidden_size, hidden_size))
        self.dense = np.random.normal(0, 0.02, (hidden_size, hidden_size))

    def transpose_for_scores(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_length, _ = x.shape
        x = x.reshape(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.transpose(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        batch_size, seq_length, _ = hidden_states.shape

        query_layer = np.matmul(hidden_states, self.query)
        key_layer = np.matmul(hidden_states, self.key)
        value_layer = np.matmul(hidden_states, self.value)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        attention_scores = np.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = np.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        context_layer = context_layer.reshape(batch_size, seq_length, self.hidden_size)

        output = np.matmul(context_layer, self.dense)

        return output

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def dropout(self, x: np.ndarray) -> np.ndarray:
        if self.attention_probs_dropout_prob > 0:
            mask = np.random.binomial(1, 1 - self.attention_probs_dropout_prob, x.shape) / (1 - self.attention_probs_dropout_prob)
            return x * mask
        return x

class FeedForward:
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob

        self.dense1 = np.random.normal(0, 0.02, (hidden_size, intermediate_size))
        self.dense2 = np.random.normal(0, 0.02, (intermediate_size, hidden_size))

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        intermediate = np.matmul(hidden_states, self.dense1)
        intermediate = self.gelu(intermediate)

        if self.hidden_dropout_prob > 0:
            intermediate = self.dropout(intermediate)

        output = np.matmul(intermediate, self.dense2)

        return output

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def dropout(self, x: np.ndarray) -> np.ndarray:
        if self.hidden_dropout_prob > 0:
            mask = np.random.binomial(1, 1 - self.hidden_dropout_prob, x.shape) / (1 - self.hidden_dropout_prob)
            return x * mask
        return x

class LayerNormalization:
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        self.hidden_size = hidden_size
        self.eps = eps
        self.gamma = np.ones(hidden_size)
        self.beta = np.zeros(hidden_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class TransformerLayer:
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
    ):
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.layer_norm1 = LayerNormalization(hidden_size)
        self.layer_norm2 = LayerNormalization(hidden_size)

    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        attention_output = self.attention.forward(hidden_states, attention_mask)
        attention_output = self.layer_norm1.forward(hidden_states + attention_output)

        feed_forward_output = self.feed_forward.forward(attention_output)
        output = self.layer_norm2.forward(attention_output + feed_forward_output)

        return output
