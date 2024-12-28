# Clinical Concept Identification in Medical Notes

## Overview
This project implements an automated system for identifying specific clinical concepts in patient notes using a modified RoBERTa model. The system helps address the challenge of scoring medical student patient notes by automatically mapping clinical concepts from exam rubrics to their various expressions in clinical documentation.

## Problem Description
Medical licensing exams require students to write patient notes that are traditionally scored by trained physicians using rubrics. This manual process is time-intensive and resource-heavy. While NLP approaches exist, the challenge lies in identifying clinical concepts that can be expressed in multiple ways (e.g., "loss of interest in activities" → "no longer plays tennis") and handling complex cases like concept mapping across multiple text segments and ambiguous negations.

## Solution
The solution implements a hybrid RoBERTa model for token classification to identify and locate clinical concepts in patient notes. Key features include:

- Custom RoBERTa-based architecture for token classification
- 5-fold cross-validation strategy
- Support for handling long sequences with truncation
- Special text processing for medical features
- Offset mapping for precise concept location identification

## Requirements
```
torch
pandas
numpy
transformers
datasets
```

## Model Architecture
The `HybridRoberta` class extends `RobertaPreTrainedModel` with:
- Base RoBERTa encoder
- Dropout layer for regularization
- Linear classifier layer
- BCE loss with logits for training
- Support for token classification outputs

## Data Processing
The pipeline includes:
1. Loading and merging patient notes, features, and test data
2. Processing feature text (handling special characters and formatting)
3. Tokenization with offset mapping for location tracking
4. Dataset creation using HuggingFace's Dataset class

## Inference Pipeline
The inference process includes:
1. Loading pre-trained models for each fold
2. Generating predictions using ensemble averaging
3. Post-processing predictions to extract concept locations
4. Converting predictions to submission format

## Usage
1. Ensure all required data files are in the correct input directory:
   - test.csv
   - patient_notes.csv
   - features.csv

2. Run the inference pipeline:
```python
# Load and prepare data
test_df = pd.read_csv("../input/nbme-score-clinical-patient-notes/test.csv")
notes_df = pd.read_csv("../input/nbme-score-clinical-patient-notes/patient_notes.csv")
feats_df = pd.read_csv("../input/nbme-score-clinical-patient-notes/features.csv")

# Run inference
# ... (follow main script)

# Generate submission
submission_df.to_csv("submission.csv", index=False)
```

## Model Configuration
Key configurations include:
- Number of folds: 5
- Max sequence length: 416
- Evaluation batch size: 64
- Dataloader workers: 2

## Output Format
The model generates predictions in the format:
```
id,location
[case_id],"[start_idx] [end_idx]; [start_idx] [end_idx]; ..."
```

## Acknowledgments
This project is based on the NBME - Score Clinical Patient Notes competition. Special thanks to:
- National Board of Medical Examiners® (NBME®)
- Dr Le An Ha from the University of Wolverhampton's Research Group in Computational Linguistics
- HuggingFace's Transformers library
