# SIP: System Initiative Prediction

**Multi-class ambiguity detection (2-4 classes) with detailed metadata output.**

## 🎯 Features

✅ **Configurable Classes**: 2, 3, or 4 ambiguity levels  
✅ **6 Baselines**: All 6 CRF variants (VanillaCRF, Who2WhoCRF, PositionCRF, Who2Who_PositionCRF, IntimeCRF, DistanceCRF)  
✅ **Rich Output**: Confidence, conditions, token importance, metadata  
✅ **F1/Precision/Recall**: Full evaluation metrics  
✅ **Training Plots**: Matplotlib visualization  

## 📊 Input Format

```json
{
  "prompt": "user instruction",
  "context": "background context",
  "can_retrieve": bool,
  "tools": "tool descriptions, req_clarification(2-4)",
  "ambiguous": "true/false",
  "ambiguous_type": 0-3,
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "function_call", "value": "..."},
    {"from": "observation", "value": "..."},
    {"from": "gpt", "value": "... or req_clarification(N)"}
  ],
  "system": "system prompt"
}
```

**Label Sources**:
1. `ambiguous_type` field (0-3) - Ground truth
2. `req_clarification(N)` in gpt response - Explicit label
3. `ambiguous` field - Binary fallback

## 📤 Output Format

```json
{
  "ambiguous_utterance": bool,
  "ambiguous_class": "clear/slightly_ambiguous/needs_clarification/highly_ambiguous",
  "total_candidates": int,
  "explanation": "Why ambiguous with confidence score",
  "conditions": [
    {"condition": "Interpretation 1", "answer": "Possible answer 1"},
    {"condition": "Interpretation 2", "answer": "Possible answer 2"}
  ],
  "metadata": {
    "confidence_score": 0.85,
    "all_predictions": [0, 1, 2],
    "all_confidences": [0.95, 0.78, 0.85],
    "num_turns": 3,
    "num_classes": 4,
    "class_distribution": {"clear": 1, "slightly_ambiguous": 1, ...},
    "token_importance": {"Book": 0.3, "it": 0.9, ...},
    "CRF_layer": {},
    "input_metadata": {"prompt": "...", "context": "...", ...}
  }
}
```

## 🏗️ Class Configurations

### 2 Classes (Binary)
- `0`: Clear
- `1`: Ambiguous

### 3 Classes  
- `0`: Clear
- `1`: Needs clarification
- `2`: Highly ambiguous

### 4 Classes (Full)
- `0`: Clear
- `1`: Slightly ambiguous
- `2`: Needs clarification  
- `3`: Highly ambiguous

## 🚀 Quick Start

### Installation
```bash
pip install torch transformers tqdm scikit-learn matplotlib
```

## 📊 Data Statistics Script

### Usage

```bash
# Single dataset
python scripts/bilstmcrf_data_statistics.py \
    --input data/sip_train.json \
    --names "Training Set" \
    --output_dir statistics

# Multiple datasets (train + test)
python scripts/bilstmcrf_data_statistics.py \
    --input data/sip_train.json data/sip_test.json \
    --names "Train" "Test" \
    --output_dir statistics
```

### Training

```bash
python scripts/train_bilstm_crf.py \
    --train_data data/sip_train.json \
    --val_split 0.15 \
    --baseline distance \
    --num_classes 4 \
    --optimizer adam \
    --epochs 30 \
    --lr 1e-3
```
```bash
python scripts/train_bilstm_crf.py \
    --train_data data/sip_train.json \
    --val_split 0.15 \
    --baseline distance \
    --num_classes 4 \
    --optimizer adam \
    --epochs 30 \
    --lr 1e-3
```

Compare different optimizers:

```bash
# Adam (adaptive learning rate, good default)
python scripts/train_enhanced.py --optimizer adam

# Adamax (variant of Adam, more stable)
python scripts/train_enhanced.py --optimizer adamax

# RMSprop (good for RNNs)
python scripts/train_enhanced.py --optimizer rmsprop

# SGD (with momentum=0.9)
python scripts/train_enhanced.py --optimizer sgd --lr 0.01
```


### Inference

```bash
python scripts/inference_bilstm_crf.py \
    --input_data data/sample.json \
    --checkpoint checkpoints/best_f1_model.pt \
    --baseline vanillacrf \
    --num_classes 2 \
    --output results/predictions.json
```

### Evaluate with Variance Analysis

```bash
python scripts/evaluate_bilstm_crf.py \
    --test_data data/sip_test.json \
    --checkpoint_dir checkpoints \
    --output_dir evaluation_final \
    --num_classes 4 \
    --seeds 42 123 456 789 1000
```
## 📋 5 Baselines

### 1. VanillaCRF
- Simple CRF with global transition matrix
- Fastest training
- Good baseline performance

### 2. VanillaCRF+features
- Embeddings for position and speaker features
- Better context modeling
- Slightly slower

### 3. DynamicCRF
- Dynamic transitions from adjacent observations
- Context-dependent modeling
- Best for varying conversation patterns

### 4. CtxPred
- Simple BERT classifier (no CRF)
- No sequence modeling
- Fast inference
- Lower performance on sequential data

### 5. MuSIc (Full)
- Complete model with all features
- Prior-Posterior framework
- Best performance
- Slowest training

!git clone -b Bi-LSTM-CRF-Baseline https://github.com/importrayhan/convqa_eval
cd /content/convqa_eval
cd convqa_eval
!python scripts/transform_convqa_to_sip.py \
    --input data/convqa_sample.json \
    --output data/sip_format2.json

## 📁 BiLSTM_CRF Baseline File Structure

```
convqa_eval/
├── model/
│   ├── bilstm_crf/
│         ├── music_baselines_complete.py            
│         ├── preprocessor.py             
│         └── output_generator.py
│
├── scripts/
│   ├── train_bilstm_crf.py      
│   ├── compute_statistics.py             
│   ├── evaluate_bilstm_crf.py      
│   ├── train_bilstm_crf.py
│   └── inference_bilstm_crf.py
│
└── data/
    ├── pacific_train.json
    ├── pacific_test.json
    ├──  
    └──  
```
## 🎓 Model Architecture

```
Input → BERT Encoder → BiLSTM (Prior/Posterior) → CRF → Predictions
                                                        ↓
                                                   Confidence
                                                   Conditions
                                                   Explanation
                                                   Metadata
```

## 📊 Evaluation Metrics

Computed metrics:
- **Accuracy**: Overall correctness
- **Precision**: Correct positives / All positives
- **Recall**: Correct positives / All actual positives  
- **F1 Score**: Harmonic mean of P and R (weighted for multiclass)
- **Confusion Matrix**: Per-class breakdown

## 💡 Usage Examples

### Binary Classification
```python
from model.preprocessor import SIPPreprocessor
from model.music_baselines import create_model

preprocessor = SIPPreprocessor(num_classes=2)
model = create_model('vanillacrf', num_classes=2)

# Train/inference...
```

### 4-Class Multiclass
```python
preprocessor = SIPPreprocessor(num_classes=4)
model = create_model('music', num_classes=4)

# Classes: clear, slightly_ambiguous, needs_clarification, highly_ambiguous
```

### Custom Output Processing
```python
from model.output_generator import SIPOutputGenerator

generator = SIPOutputGenerator(preprocessor, class_names)
output = generator.generate_output(data, predictions, confidences)

# Access metadata
print(output['metadata']['confidence_score'])
print(output['metadata']['token_importance'])
```

## 🔧 Hyperparameters

```python
# Model
hidden_size=256      # BiLSTM hidden size
num_layers=2         # BiLSTM layers
dropout=0.5          # Dropout rate
lambda_mle=0.1       # MLE loss weight

# Training
epochs=10-20         # Training epochs
lr=1e-3              # Learning rate (Adam)
```



## 🎯 Key Components

### Preprocessor
- Parses custom input format
- Extracts labels from multiple sources
- BERT tokenization
- Configurable class counts

### Model Baselines
- All in one file (`music_baselines.py`)
- Factory function: `create_model(baseline, num_classes)`
- Consistent interface


## 📚 Citation

```bibtex
@inproceedings{meng2023system,
  title={System initiative prediction for multi-turn conversational information seeking},
  author={Meng, Chuan and Aliannejadi, Mohammad and de Rijke, Maarten},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={1807--1817},
  year={2023}
}
```

