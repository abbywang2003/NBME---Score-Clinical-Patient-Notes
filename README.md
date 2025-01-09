# NBME Clinical Patient Notes Scoring 🏥

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)

## 📋 Project Overview

This project implements a RoBERTa-based model for automatically scoring clinical patient notes from the NBME competition. The model identifies and extracts relevant medical concepts from clinical patient notes using token classification.

## 🔧 Model Architecture

The implementation uses a hybrid RoBERTa architecture with:
- Base RoBERTa model for token embeddings
- Custom classification head
- Token-level binary classification
- BCE loss with masked selection

```python
class HybridRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
```

## 🛠️ Setup & Dependencies

### Requirements
```txt
torch
pandas
numpy
transformers
datasets
```

### Installation
```bash
pip install torch pandas numpy transformers datasets
```

## 📁 Data Structure

The model expects three CSV files:
- `test.csv`: Test set data
- `patient_notes.csv`: Patient clinical notes
- `features.csv`: Feature descriptions

## 🚀 Usage

1. Configure settings in `CFG` class:
```python
class CFG:
    n_folds = 5
    model_path = "../input/5f-rob-b-nbme/fold{fold}"
```

2. Load and preprocess data:
```python
test_df = pd.read_csv("../input/nbme-score-clinical-patient-notes/test.csv")
notes_df = pd.read_csv("../input/nbme-score-clinical-patient-notes/patient_notes.csv")
feats_df = pd.read_csv("../input/nbme-score-clinical-patient-notes/features.csv")
```

3. Run predictions:
```python
predictions = trainer.predict(tokenized_ds)
```

## 🔄 Data Processing Pipeline

1. **Feature Text Processing**
   - Replaces special characters
   - Standardizes formatting

2. **Tokenization**
   - Maximum length: 416 tokens
   - Returns offset mapping
   - Truncation on second sequence

3. **Prediction Processing**
   - Applies sigmoid activation
   - Converts to location format
   - Joins multiple predictions

## 📊 Model Training

The model uses 5-fold cross-validation with:
- Batch size: 64
- Multiple workers for data loading
- Token classification approach

## 🔍 Prediction Format

Output predictions are formatted as:
```python
{
    "id": [note_ids],
    "location": ["start_idx end_idx; start_idx end_idx"]
}
```

## 💾 Save Predictions

Results are saved to `submission.csv`:
```python
submission_df = pd.DataFrame(data={
    "id": tokenized_ds["id"], 
    "location": location_predictions
})
submission_df.to_csv("submission.csv", index=False)
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on Hugging Face's Transformers library
- RoBERTa model architecture
- NBME competition dataset
