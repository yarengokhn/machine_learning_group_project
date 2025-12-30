import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model import HybridEncoder, HybridDecoder, Seq2Seq
from src.data_loader import tokenized_dataset, tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in iterator:   # ✅ ВОТ ЭТОГО НЕ ХВАТАЛО

            # batch может быть dict (HuggingFace) или tuple
            if isinstance(batch, dict):
                src = batch["input_ids"].to(DEVICE)
                trg = batch["labels"].to(DEVICE)
            else:
                src, trg = batch
                src = src.to(DEVICE)
                trg = trg.to(DEVICE)

            output = model(src, trg)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def main():
    # 2) Build model exactly like in train.py
    INPUT_DIM = tokenizer.vocab_size
    OUTPUT_DIM = tokenizer.vocab_size
    EMB_DIM = 128
    HID_DIM = 256

    enc = HybridEncoder(INPUT_DIM, EMB_DIM, HID_DIM)
    dec = HybridDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    # 3) Load weights
    model.load_state_dict(torch.load("models/model.pt", map_location=DEVICE))

    # 4) DataLoader (use smaller batch if CPU)
    test_iterator = DataLoader(tokenized_dataset, batch_size=32, shuffle=False)

    # 5) Loss
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    test_loss = evaluate(model, test_iterator, criterion)
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
