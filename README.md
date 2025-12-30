# Code Summarization Project

This project implements a sequence-to-sequence model with attention to generate natural language descriptions of Python code snippets.

## Architecture

Data → Representation → Model → Training → Evaluation → Inference

- **Encoder**: Bidirectional GRU
- **Attention**: Bahdanau Attention
- **Decoder**: GRU with attention
- **Metrics**: BLEU, ROUGE, Perplexity

## Project Structure

```
project/
 ├── src/
 │   ├── data/
 │   │   ├── dataset.py
 │   │   └── preprocess.py
 │   ├── models/
 │   │   ├── encoder.py
 │   │   ├── decoder.py
 │   │   ├── attention.py
 │   │   └── seq2seq.py
 │   ├── training/
 │   │   ├── train_loop.py
 │   │   └── evaluate.py
 │   └── inference.py
 ├── scripts/
 │   ├── train.py
 │   ├── evaluate.py
 │   └── summarize.py
 ├── checkpoints/
 ├── data/
 └── README.md
```

## Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Training

To train the model (it will auto-generate dummy data if no CSV is found):

```bash
python scripts/train.py --epochs 5 --batch_size 16
```

To use CodeSearchNet (requires `datasets` and `pyarrow`):

```bash
python scripts/train.py --use_codesearchnet --csn_limit 5000 --epochs 10
```

### 3. Evaluation

```bash
python scripts/evaluate.py --test_data data/test.csv
```

### 4. Summarization (Inference)

```bash
python scripts/summarize.py --input "def add(a, b): return a + b"
```

## Reproducibility

Fixed random seeds and modular components ensure stability and reproducibility following the course guidelines.