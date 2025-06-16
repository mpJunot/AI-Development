import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

from core.bert import BERT
from core.tokenizer import BertTokenizer
from core.finetuning import SentimentClassifier

EVAL_DIR = 'eval'
os.makedirs(EVAL_DIR, exist_ok=True)

def plot_training_metrics(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_metrics_distribution(metrics, save_path=None):
    plt.figure(figsize=(10, 6))
    metrics_values = list(metrics.values())
    metrics_names = list(metrics.keys())
    plt.bar(metrics_names, metrics_values)
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_score, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_predictions(texts, predictions, probabilities, save_path=None):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.bar(['Positive', 'Negative'],
            [np.mean(predictions), 1 - np.mean(predictions)])
    plt.title('Prediction Distribution')
    plt.ylabel('Proportion')

    plt.subplot(2, 1, 2)
    y_pos = np.arange(len(texts))
    plt.barh(y_pos, probabilities)
    plt.yticks(y_pos, [f'Text {i+1}' for i in range(len(texts))])
    plt.xlabel('Confidence Score')
    plt.title('Prediction Confidence by Text')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def load_sample_data():
    texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The service at this restaurant was terrible and the food was cold.",
        "I had a great experience with this product, highly recommended!",
        "The customer support was unhelpful and rude.",
        "This book is a masterpiece, couldn't put it down!",
        "The hotel room was dirty and the staff was unfriendly.",
        "The concert was amazing, the band played perfectly!",
        "The software is buggy and keeps crashing.",
        "The vacation was wonderful, beautiful location!",
        "The delivery was late and the package was damaged."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return texts, np.array(labels)

def make_predictions(classifier, tokenizer, texts):
    encoded = tokenizer.encode(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
    )

    input_ids = np.array(encoded["input_ids"])
    attention_mask = np.array(encoded["attention_mask"])

    predictions, probabilities = classifier.predict(input_ids, attention_mask)
    return predictions, probabilities

def main():
    tokenizer = BertTokenizer(do_lower_case=True)
    bert = BERT(vocab_size=len(tokenizer.vocab))
    classifier = SentimentClassifier(bert)

    texts, labels = load_sample_data()

    encoded = tokenizer.encode(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
    )

    input_ids = np.array(encoded["input_ids"])
    attention_mask = np.array(encoded["attention_mask"])

    train_losses = []
    val_losses = []

    for epoch in range(10):
        loss = classifier.train_step(input_ids, labels, attention_mask)
        train_losses.append(loss)

        val_loss = classifier.train_step(input_ids, labels, attention_mask)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/10 - Loss: {loss:.4f}")

    plot_training_metrics(
        train_losses,
        val_losses,
        save_path=os.path.join(EVAL_DIR, 'training_metrics.png')
    )

    metrics = classifier.evaluate(input_ids, labels, attention_mask)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    plot_metrics_distribution(
        metrics,
        save_path=os.path.join(EVAL_DIR, 'metrics_distribution.png')
    )

    cm = metrics['confusion_matrix']
    plot_confusion_matrix(
        cm,
        save_path=os.path.join(EVAL_DIR, 'confusion_matrix.png')
    )

    y_score = classifier.predict(input_ids, attention_mask)[1]
    plot_roc_curve(
        labels,
        y_score,
        save_path=os.path.join(EVAL_DIR, 'roc_curve.png')
    )

    new_texts = [
        "This product exceeded my expectations in every way!",
        "I'm extremely disappointed with the quality of service.",
        "The new features are amazing and very useful.",
        "The interface is confusing and poorly designed."
    ]

    predictions, probabilities = make_predictions(classifier, tokenizer, new_texts)

    print("\nPredictions for new texts:")
    for text, pred, prob in zip(new_texts, predictions, probabilities):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"Text: {text}")
        print(f"Prediction: {sentiment}")
        print(f"Confidence: {prob:.2%}\n")

    plot_predictions(
        new_texts,
        predictions,
        probabilities,
        save_path=os.path.join(EVAL_DIR, 'predictions.png')
    )

if __name__ == "__main__":
    main()
