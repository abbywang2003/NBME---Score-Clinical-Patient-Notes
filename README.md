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

## 📊 Dataset Description

The dataset consists of clinical patient notes and their annotations from the NBME competition. It includes approximately 40,000 patient notes with feature annotations for medical concept extraction.

### Data Files

#### 1. Patient Notes (`patient_notes.csv`)
- Collection of ~40,000 patient history records
- **Fields:**
  - `pn_num`: Unique identifier for each note
  - `case_num`: Clinical case identifier
  - `pn_history`: Full text of patient encounter
- Notes in test set are excluded from public version
- Suitable for unsupervised learning on unannotated notes

#### 2. Features (`features.csv`)
Clinical case rubrics containing key medical concepts
- **Fields:**
  - `feature_num`: Unique feature identifier
  - `case_num`: Case identifier
  - `feature_text`: Detailed feature description

#### 3. Training Data (`train.csv`)
Annotated subset of 1,000 patient notes
- 100 notes for each of 10 clinical cases
- **Fields:**
  - `id`: Unique identifier for note/feature pair
  - `pn_num`: Patient note reference
  - `feature_num`: Feature reference
  - `case_num`: Clinical case identifier
  - `annotation`: Feature text instances
  - `location`: Character spans of annotations
    - Multiple spans separated by semicolons
    - Format: "start_idx end_idx; start_idx end_idx"

#### 4. Test Data
- Contains ~2,000 patient notes
- Uses same clinical cases as training set
- Added to `patient_notes.csv` during evaluation
- Format matches training data structure

### Data Structure Example

```python
# Patient Note Example
{
    "pn_num": "100",
    "case_num": "1",
    "pn_history": "Patient presents with..."
}

# Feature Example
{
    "feature_num": "1",
    "case_num": "1",
    "feature_text": "shortness of breath"
}

# Annotation Example
{
    "id": "1",
    "pn_num": "100",
    "feature_num": "1",
    "case_num": "1",
    "annotation": "dyspnea",
    "location": "45 51; 67 89"
}
```

### Dataset Statistics
- Total Patient Notes: 42,126
- Annotated Notes: 1,000
- Clinical Cases: 10
- Test Set Size: ~2,000 notes
- Features per Case: 1

### Data Preprocessing
1. **Text Cleaning**
   ```python
   def process_feature_text(text):
       return text.replace("-OR-", ";-").replace("-", " ")
   ```

2. **Tokenization**
   ```python
   tokenizer = AutoTokenizer.from_pretrained(CFG.model_path.format(fold=0))
   max_length = 416
   truncation = "only_second"
   ```

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

## 🙏 Acknowledgments

- Based on Hugging Face's Transformers library
- RoBERTa model architecture
- NBME competition dataset
