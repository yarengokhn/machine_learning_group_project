import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer
from src.preprocessing import preprocess_code

LIMIT_TRAIN_SAMPLES = 200

dataset = load_dataset(
    "code_x_glue_ct_code_to_text",
    "python",
    split="train"
)

dataset = dataset.select(
    range(min(LIMIT_TRAIN_SAMPLES, len(dataset)))
)

if "docstring" in dataset.column_names and "summary" not in dataset.column_names:
    dataset = dataset.rename_column("docstring", "summary")

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(examples):
    # Temizlik adımlarını uygula
    cleaned_codes = [preprocess_code(c, is_Code=True) for c in examples["code"]]
    cleaned_summaries = [preprocess_code(s, is_Code=False) for s in examples["summary"]]
    
    # Kodu sayılara çevir (Encoder için) [cite: 49]
    model_inputs = tokenizer(cleaned_codes, padding="max_length", truncation=True, max_length=256)
    
    # Özeti sayılara çevir (Decoder hedefi için) [cite: 49]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(cleaned_summaries, padding="max_length", truncation=True, max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Veri setine uygula 
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# после tokenized_dataset = dataset.map(...)

# оставляем только нужные поля (если есть лишнее — можно не удалять, но лучше чисто)
keep_cols = [c for c in ["input_ids", "attention_mask", "labels"] if c in tokenized_dataset.column_names]

tokenized_dataset.set_format(type="torch", columns=keep_cols)