import numpy as np
from typing import List, Dict, Tuple, Optional
from .bert import BERT
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentClassifier:
    def __init__(
        self,
        bert: BERT,
        num_labels: int = 2,
        learning_rate: float = 2e-5,
    ):
        self.bert = bert
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.classifier = np.random.normal(0, 0.02, (bert.hidden_size, num_labels))

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass for sentiment classification."""
        _, pooled_output = self.bert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = np.matmul(pooled_output, self.classifier)
        return logits

    def compute_loss(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute cross-entropy loss."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        n_samples = len(labels)
        loss = -np.sum(np.log(probs[np.arange(n_samples), labels])) / n_samples
        return loss

    def train_step(
        self,
        input_ids: np.ndarray,
        labels: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> float:
        """Perform one training step."""
        logits = self.forward(input_ids, attention_mask)

        loss = self.compute_loss(logits, labels)

        grad = self._compute_gradients(logits, labels)
        self._update_weights(grad)

        return loss

    def predict(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get predictions."""
        logits = self.forward(input_ids, attention_mask)
        return np.argmax(logits, axis=-1)

    def evaluate(
        self,
        input_ids: np.ndarray,
        labels: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate the model."""
        predictions = self.predict(input_ids, attention_mask)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        plt.savefig('metrics.png')
        plt.close()

        return metrics

    def _compute_gradients(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients (simplified version)."""
        grad = {}

        grad["classifier"] = np.zeros_like(self.classifier)

        grad["bert"] = {}
        for param_name, param in self.bert.__dict__.items():
            if isinstance(param, np.ndarray):
                grad["bert"][param_name] = np.zeros_like(param)

        return grad

    def _update_weights(self, grad: Dict[str, np.ndarray]):
        """Update weights using gradients."""
        self.classifier -= self.learning_rate * grad["classifier"]

        for param_name, param_grad in grad["bert"].items():
            if hasattr(self.bert, param_name):
                param = getattr(self.bert, param_name)
                if isinstance(param, np.ndarray):
                    param -= self.learning_rate * param_grad
