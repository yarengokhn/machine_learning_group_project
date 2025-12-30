import re
import pandas as pd
import collections
import pickle
import os
from datasets import load_dataset

class Tokenizer:
    def __init__(self, is_code=True):
        self.is_code = is_code

    def tokenize(self, text):
        if not isinstance(text, str):
            return []
        
        # Lowercase
        text = text.lower()
        
        if self.is_code:
            # Simple whitespace and punctuation split for code
            # In a real scenario, we might use the 'tokenize' module or more complex regex
            tokens = re.findall(r"[\w']+|[^\w\s]", text)
        else:
            # Word-level tokenizer for summaries
            tokens = re.findall(r"[\w']+|[^\w\s]", text)
        
        return tokens

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freqs = collections.Counter()

    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            for word in sentence:
                self.freqs[word] += 1
        
        idx = 4
        for word, freq in self.freqs.items():
            if freq >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len=None):
        ids = [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        return [self.itos.get(idx, "<unk>") for idx in ids]

def preprocess_data(df, code_col="code", summary_col="summary", max_code_len=100, max_summary_len=30):
    # Remove empty or extremely short summaries
    df = df[df[summary_col].str.len() > 5].copy()
    
    # Normalize whitespace
    df[code_col] = df[code_col].apply(lambda x: " ".join(x.split()))
    df[summary_col] = df[summary_col].apply(lambda x: " ".join(x.split()).lower())
    
    return df

def save_vocab(vocab, path):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_codesearchnet_dataset(split="train", limit=1000):
    """
    Loads CodeSearchNet dataset (via CodeXGlue) for Python using Hugging Face 'datasets'
    """
    print(f"Loading CodeSearchNet dataset (CodeXGlue, {split} split, limit={limit})...")
    # Using code_x_glue_ct_code_to_text as it is a reliable mirror of CSN
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split=f"{split}[:{limit}]")
    
    data = []
    for item in dataset:
        # CodeXGlue items have 'code' and 'docstring'
        code = item.get('code', '')
        summary = item.get('docstring', '')
        if code and summary:
            data.append({"code": code, "summary": summary})
            
    return pd.DataFrame(data)
