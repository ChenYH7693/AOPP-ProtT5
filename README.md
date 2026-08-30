AOPP-ProtT5
Multi-Scale Pooling and Residual Refinement with ProtT5 Embeddings for Antioxidant Peptide Prediction
This repository contains the implementation of AOPP-ProtT5: Multi-Scale Pooling and Residual Refinement with ProtT5 Embeddings for Antioxidant Peptide Prediction, a ProtT5-based framework for antioxidant peptide (AOP) prediction.
The project uses the pretrained `Rostlab/prot_t5_xl_uniref50` encoder and supports configurable fine-tuning, threshold optimization, early stopping, EMA, focal loss, AMP, and evaluation visualization.
Project Structure
```text
AOPP-ProtT5/
├── AOP.py
└── AOP_data/
    ├── AOPP/
    ├── AnOxPP/
    ├── AnOxPePred/
    └── combined/
```
`AOP.py` is the main training script.
Features
ProtT5-based peptide sequence representation
Binary antioxidant peptide classification
Stratified K-fold cross-validation
Configurable partial encoder fine-tuning
Separate learning rates for encoder and classification head
Automatic decision-threshold search
Early stopping
Exponential Moving Average (EMA)
Optional focal loss
Optional mixed-precision training (AMP)
Optional gradient checkpointing
ROC, PR, confusion-matrix, loss, accuracy, and MCC plots
Per-fold predictions and summary metrics
Requirements
Python 3.9+ is recommended.
Install the main dependencies:
```bash
pip install torch transformers sentencepiece pandas numpy scikit-learn matplotlib tqdm
```
For GPU training, install a PyTorch build compatible with your CUDA version.
Dataset Format
Input data should be a CSV file containing:
```csv
SEQUENCE,label
ACDEFGHIK,1
LMNPQRSTV,0
```
Required columns:
`SEQUENCE`: amino-acid sequence
`label`: binary class label (`0` or `1`)
The script also accepts `labe` as the label column name.
During preprocessing:
sequences are converted to uppercase;
non-alphabetic characters are removed;
`U`, `Z`, `O`, and `B` are replaced with `X`;
empty sequences are removed.
The default dataset path is:
```text
AOP_data/AnOxPP/AnOxPP.csv
```
Usage
Run the default 5-fold experiment:
```bash
python AOP.py
```
Specify your own dataset and output directory:
```bash
python AOP.py \
  --csv_path AOP_data/AnOxPP/AnOxPP.csv \
  --output_dir outputs \
  --epochs 10 \
  --batch_size 4 \
  --max_len 256 \
  --lr_head 2e-4 \
  --lr_encoder 8e-6 \
  --unfreeze_last_n_layers 2
```
Example with additional training options:
```bash
python AOP.py \
  --csv_path AOP_data/AnOxPP/AnOxPP.csv \
  --output_dir outputs \
  --n_splits 5 \
  --epochs 10 \
  --batch_size 4 \
  --use_focal_loss \
  --gradient_checkpointing \
  --select_metric Accuracy
```
Main Arguments
Argument	Default	Description
`--csv_path`	`AOP_data/AnOxPP/AnOxPP.csv`	Input CSV path
`--output_dir`	`AOP-outputs-accuracy-boost-AnOxPP`	Directory for results
`--model_name`	`Rostlab/prot_t5_xl_uniref50`	Hugging Face model
`--n_splits`	`5`	Number of CV folds
`--epochs`	`10`	Maximum epochs per fold
`--batch_size`	`4`	Batch size
`--num_workers`	`2`	DataLoader workers
`--max_len`	`256`	Maximum tokenized sequence length
`--lr_head`	`2e-4`	Learning rate for classification head
`--lr_encoder`	`8e-6`	Learning rate for unfrozen encoder layers
`--weight_decay`	`1e-2`	AdamW weight decay
`--warmup_ratio`	`0.1`	Scheduler warmup ratio
`--early_stopping_patience`	`4`	Early-stopping patience
`--seed`	`42`	Random seed
`--dropout`	`0.35`	Classification-head dropout
`--unfreeze_last_n_layers`	`2`	Number of final ProtT5 layers to fine-tune
`--grad_accum_steps`	`1`	Gradient accumulation steps
`--max_grad_norm`	`1.0`	Gradient clipping norm
`--ema_decay`	`0.999`	EMA decay
`--use_focal_loss`	off	Enable focal loss
`--focal_alpha`	`0.25`	Focal-loss alpha
`--focal_gamma`	`2.0`	Focal-loss gamma
`--select_metric`	`Accuracy`	Metric used for model/threshold selection
`--gradient_checkpointing`	off	Enable gradient checkpointing
`--no_amp`	off	Disable automatic mixed precision
Available values for `--select_metric` are:
```text
Accuracy, MCC, Precision, Sensitivity, Specificity
```
Evaluation Metrics
The pipeline reports:
Accuracy
MCC (Matthews Correlation Coefficient)
Precision
Sensitivity
Specificity
Validation loss
ROC-AUC
Average Precision / PR curve
For each validation fold, the classification threshold is searched over a range of candidate values and selected according to `--select_metric`.
Outputs
The output directory contains the cleaned dataset and experiment configuration, together with per-fold and cross-validation results.
Typical outputs include:
```text
outputs/
├── cleaned_dataset.csv
├── config.json
├── cv_results.csv
├── cv_summary.json
├── cv_fold_metrics.png
├── cv_mean_metrics.png
└── fold_*/
    ├── val_predictions.csv
    ├── fold_result.json
    ├── fold_*_loss_curve.png
    ├── fold_*_accuracy_curve.png
    ├── fold_*_mcc_curve.png
    ├── fold_*_confusion_matrix.png
    ├── fold_*_roc_curve.png
    └── fold_*_pr_curve.png
```
Exact files may depend on the training configuration.
Notes
ProtT5 is a large protein language model and can require substantial GPU memory. If you encounter out-of-memory errors, consider:
```bash
python AOP.py \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --gradient_checkpointing
```
You can also reduce `--max_len` or fine-tune fewer encoder layers with `--unfreeze_last_n_layers`.
Reproducibility
The default random seed is `42`. The same seed is used for Python, NumPy, PyTorch, and stratified cross-validation splitting.
License
No license has been specified for this repository yet. Add a `LICENSE` file if you plan to distribute or reuse the project publicly.
Acknowledgements
This project uses the ProtT5 protein language model from Rostlab and the Hugging Face Transformers ecosystem.
