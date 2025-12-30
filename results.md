# Model Training and Evaluation Results

## Training Summary
- **Model**: Seq2Seq with Attention
- **Dataset**: CodeSearchNet (Python, 1000 samples)
- **Epochs**: 10
- **Final Train Loss**: 1.494
- **Final Val Loss**: 3.558

## Evaluation Metrics (Test Set, 50 samples)
- **BLEU Score**: 0.0046
- **ROUGE-1 (F1)**: 0.2309
- **ROUGE-2 (F1)**: 0.0632
- **ROUGE-L (F1)**: 0.2288

### Qualitative Samples
1. **Reference**: `str->list Convert XML to URL List. From Biligrab.`
   **Generated**: `convert - > list list to to <unk> .`
2. **Reference**: `Downloads Dailymotion videos by URL.`
   **Generated**: `<unk> <unk> by by <unk> .`
3. **Reference**: `Print a log message to standard error.`
   **Generated**: `appends a local message to to .`

## How to Reproduce
Run the following commands:
```bash
python3 scripts/train.py --config configs/base.yaml
python3 scripts/evaluate.py --config configs/base.yaml
### Round 2: Scaling Up (In Progress)
- **Config**: `configs/improved.yaml`
- **Dataset Size**: 5,000 samples
- **Model Name**: `improved_v1`
- **Status**: Training started.

```bash
python3 scripts/train.py --config configs/improved.yaml
```
summary - round 1 training 
We implemented a Seq2Seq model with attention and trained it on a subset of the CodeSearchNet Python dataset (1000 training samples). The model achieves a final training loss of 1.494 and validation loss of 3.558. On a small test set (50 samples), we obtain BLEU = 0.0046, ROUGE-1 F1 = 0.2309, ROUGE-2 F1 = 0.0632, and ROUGE-L F1 = 0.2288.
Qualitative inspection shows that the baseline tends to generate repetitive phrases and <unk> tokens, indicating that the training subset is too small and the vocabulary is too limited. In the next step, we increase the training data size and vocabulary and refine tokenization.”