# BERT Implementation from Scratch

Pure NumPy implementation of BERT (Bidirectional Encoder Representations from Transformers) with sentiment analysis fine-tuning.

## Features

- Pure NumPy implementation (no deep learning frameworks)
- Complete BERT architecture
- WordPiece tokenization
- Pre-training with Masked Language Modeling (MLM)
- Fine-tuning for sentiment analysis
- Comprehensive evaluation metrics and visualizations

## Project Structure

```
BERT/
├── core/
│   ├── bert.py           # BERT model implementation
│   ├── transformer.py    # Transformer components
│   ├── tokenizer.py      # BERT tokenizer
│   ├── pretraining.py    # Pre-training logic
│   └── finetuning.py     # Fine-tuning for sentiment analysis
├── eval/                 # Evaluation visualizations
└── README.md
```

## Usage

### Pre-training

```python
from core.bert import BERT
from core.tokenizer import BertTokenizer
from core.pretraining import BertPretraining

tokenizer = BertTokenizer(do_lower_case=True)
bert = BERT(vocab_size=len(tokenizer.vocab))
pretraining = BertPretraining(bert, tokenizer)

loss = pretraining.train_step(input_ids, attention_mask)
```

### Fine-tuning for Sentiment Analysis

```python
from core.finetuning import SentimentClassifier

classifier = SentimentClassifier(bert)

loss = classifier.train_step(input_ids, labels, attention_mask)
metrics = classifier.evaluate(input_ids, labels, attention_mask)
predictions, probabilities = classifier.predict(new_texts)
```

## Evaluation

The implementation includes comprehensive evaluation metrics and visualizations:

- Training and validation loss curves
- Confusion matrix
- ROC curve
- Performance metrics (accuracy, precision, recall, F1)
- Prediction confidence visualization

All visualizations are saved in the `eval/` directory.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
