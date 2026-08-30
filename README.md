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
No license has been specified for this repository yet. Add a `LICENSE` file if you plan to distribute or reuse the project publicly.
Acknowledgements
This project uses the ProtT5 protein language model from Rostlab and the Hugging Face Transformers ecosystem.
