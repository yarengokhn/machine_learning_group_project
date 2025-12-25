import os
import sys

from datasets import load_dataset

# Üst dizine erişim izni (modül hatalarını engellemek için)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess_code
from transformers import AutoTokenizer

# Python veri setini indirelim
dataset = load_dataset("Nan-Do/code-search-net-python")
# Kod için en uygun pretrained tokenizer'lardan biri
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(examples):
    # Temizlik adımlarını uygula
    cleaned_codes = [preprocess_code(c, is_Code=True) for c in examples["code"]]
    cleaned_summaries = [preprocess_code(s, is_Code=False) for s in examples["summary"]]
    
    # Kodu sayılara çevir (Encoder için) [cite: 49]
    model_inputs = tokenizer(cleaned_codes, padding="max_length", truncation=True, max_length=128)
    
    # Özeti sayılara çevir (Decoder hedefi için) [cite: 49]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(cleaned_summaries, padding="max_length", truncation=True, max_length=64)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Veri setine uygula
tokenized_dataset = dataset.map(tokenize_function, batched=True)