import numpy as np
from typing import List, Dict, Tuple, Optional
from .bert import BERT
from .tokenizer import BertTokenizer

class BertPretraining:
    def __init__(
        self,
        bert: BERT,
        tokenizer: BertTokenizer,
        mlm_probability: float = 0.15,
    ):
        self.bert = bert
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_head = np.random.normal(0, 0.02, (bert.hidden_size, bert.vocab_size))

    def _create_mlm_mask(self, input_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, seq_length = input_ids.shape
        probability_matrix = np.full(input_ids.shape, self.mlm_probability)
        special_tokens_mask = np.zeros_like(input_ids, dtype=bool)
        for special_token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
            special_tokens_mask |= (input_ids == self.tokenizer.vocab[special_token])

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
        labels = np.full(input_ids.shape, -100)
        labels[masked_indices] = input_ids[masked_indices]
        input_ids[masked_indices] = self.tokenizer.vocab[self.tokenizer.mask_token]

        return input_ids, labels

    def forward(
        self,
        input_ids: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        masked_input_ids, labels = self._create_mlm_mask(input_ids)
        sequence_output, _ = self.bert.forward(
            masked_input_ids,
            token_type_ids,
            attention_mask,
        )
        mlm_logits = np.matmul(sequence_output, self.mlm_head)

        return {
            "logits": mlm_logits,
            "labels": labels,
        }

    def compute_loss(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        logits = logits.reshape(-1, self.bert.vocab_size)
        labels = labels.reshape(-1)
        mask = (labels != -100)
        logits = logits[mask]
        labels = labels[mask]
        log_probs = self._log_softmax(logits)
        nll_loss = -np.sum(log_probs[np.arange(len(labels)), labels]) / len(labels)

        return nll_loss

    def _log_softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        log_sum_exp = np.log(np.sum(exp_x, axis=-1, keepdims=True))
        return x - log_sum_exp

    def train_step(
        self,
        input_ids: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
        learning_rate: float = 1e-4,
    ) -> float:
        outputs = self.forward(input_ids, token_type_ids, attention_mask)
        loss = self.compute_loss(outputs["logits"], outputs["labels"])
        grad = self._compute_gradients(outputs["logits"], outputs["labels"])
        self._update_weights(grad, learning_rate)

        return loss

    def _compute_gradients(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        grad = {}
        grad["mlm_head"] = np.zeros_like(self.mlm_head)
        grad["bert"] = {}
        for param_name, param in self.bert.__dict__.items():
            if isinstance(param, np.ndarray):
                grad["bert"][param_name] = np.zeros_like(param)

        return grad

    def _update_weights(
        self,
        grad: Dict[str, np.ndarray],
        learning_rate: float,
    ):
        self.mlm_head -= learning_rate * grad["mlm_head"]
        for param_name, param_grad in grad["bert"].items():
            if hasattr(self.bert, param_name):
                param = getattr(self.bert, param_name)
                if isinstance(param, np.ndarray):
                    param -= learning_rate * param_grad
