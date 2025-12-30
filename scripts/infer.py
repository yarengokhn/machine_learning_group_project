# scripts/infer.py
import os
import sys
import torch
import torch.nn as nn

# чтобы работали imports вида: from models.model import ...
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.model import HybridEncoder, HybridDecoder, Seq2Seq
from src.data_loader import tokenized_dataset, tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_special_ids(tok):
    # максимально безопасно вытаскиваем спец-токены
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id

    # для Roberta/CodeBERT обычно bos=<s>, eos=</s>
    if bos_id is None:
        bos_id = tok.cls_token_id if tok.cls_token_id is not None else pad_id
    if eos_id is None:
        eos_id = tok.sep_token_id if tok.sep_token_id is not None else pad_id

    return bos_id, eos_id, pad_id


@torch.no_grad()
def greedy_generate(model, src_ids, max_new_tokens=64):
    """
    Универсальная генерация без доступа к decoder-step:
    1) начинаем с BOS
    2) каждый шаг: logits = model(src, current_trg) -> берём последний timestep -> argmax
    """
    model.eval()

    bos_id, eos_id, pad_id = get_special_ids(tokenizer)

    # src_ids: [seq] -> [1, seq]
    if torch.is_tensor(src_ids):
        src = src_ids.unsqueeze(0).to(DEVICE)
    else:
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # trg начинается с BOS: [1,1]
    trg = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        logits = model(src, trg)  # ожидаем [B, T, vocab]
        next_token_logits = logits[:, -1, :]  # последний timestep
        next_id = torch.argmax(next_token_logits, dim=-1).item()

        # добавляем токен
        trg = torch.cat([trg, torch.tensor([[next_id]], device=DEVICE)], dim=1)

        if next_id == eos_id:
            break

    # возвращаем сгенерированную последовательность без batch dim
    return trg.squeeze(0).tolist()


def decode_ids(ids):
    # декодим красиво, без спец-токенов
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    # 1) build model exactly like train.py/evaluate.py
    INPUT_DIM = tokenizer.vocab_size
    OUTPUT_DIM = tokenizer.vocab_size
    EMB_DIM = 128
    HID_DIM = 256

    enc = HybridEncoder(INPUT_DIM, EMB_DIM, HID_DIM).to(DEVICE)
    dec = HybridDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM).to(DEVICE)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    # 2) load weights
    weights_path = "models/model.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Can't find {weights_path}. Train first to create it.")

    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)

    # 3) pick a few samples
    n_samples = 5
    max_new_tokens = 64

    print(f"DEVICE: {DEVICE}")
    print(f"Running inference on {n_samples} samples...\n")

    for i in range(n_samples):
        sample = tokenized_dataset[i]

        src_ids = sample["input_ids"]
        ref_ids = sample.get("labels", None)

        pred_ids = greedy_generate(model, src_ids, max_new_tokens=max_new_tokens)

        code_text = decode_ids(src_ids.tolist() if torch.is_tensor(src_ids) else src_ids)
        pred_text = decode_ids(pred_ids)
        ref_text = decode_ids(ref_ids.tolist()) if (ref_ids is not None and torch.is_tensor(ref_ids)) else ""

        # чтобы вывод не был огромным
        code_preview = (code_text[:600] + " ...") if len(code_text) > 600 else code_text

        print("=" * 90)
        print(f"[Sample {i}] CODE (preview):\n{code_preview}\n")
        print(f"[Sample {i}] PRED SUMMARY:\n{pred_text}\n")
        if ref_text:
            print(f"[Sample {i}] REF SUMMARY:\n{ref_text}\n")

    print("=" * 90)
    print("Done.")


if __name__ == "__main__":
    main()