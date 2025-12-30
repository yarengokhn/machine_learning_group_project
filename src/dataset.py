import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CodeSummarizationDataset(Dataset):
    def __init__(self, df, code_vocab, summary_vocab, code_tokenizer, summary_tokenizer, max_code_len=100, max_summary_len=30):
        self.df = df
        self.code_vocab = code_vocab
        self.summary_vocab = summary_vocab
        self.code_tokenizer = code_tokenizer
        self.summary_tokenizer = summary_tokenizer
        self.max_code_len = max_code_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = row['code']
        summary = row['summary']

        # Tokenize
        code_tokens = self.code_tokenizer.tokenize(code)
        summary_tokens = self.summary_tokenizer.tokenize(summary)

        # Numericalize
        code_ids = self.code_vocab.encode(code_tokens, max_len=self.max_code_len)
        summary_ids = self.summary_vocab.encode(summary_tokens, max_len=self.max_summary_len - 2) # Leave space for SOS and EOS

        # Add SOS and EOS to summary
        summary_input_ids = [self.summary_vocab.stoi["<sos>"]] + summary_ids + [self.summary_vocab.stoi["<eos>"]]
        
        # summary_target_ids is shifted (for cross entropy, usually we handle this in the train loop or here)
        # Here we return the full sequence and handle shifting in the decoder/loss
        
        return {
            "code_ids": torch.tensor(code_ids, dtype=torch.long),
            "summary_ids": torch.tensor(summary_input_ids, dtype=torch.long)
        }

def collate_fn(batch, pad_idx_code, pad_idx_summary):
    code_ids = [item["code_ids"] for item in batch]
    summary_ids = [item["summary_ids"] for item in batch]

    code_ids_padded = pad_sequence(code_ids, batch_first=True, padding_value=pad_idx_code)
    summary_ids_padded = pad_sequence(summary_ids, batch_first=True, padding_value=pad_idx_summary)

    return {
        "code_ids": code_ids_padded,
        "summary_ids": summary_ids_padded
    }

def get_dataloader(df, code_vocab, summary_vocab, code_tokenizer, summary_tokenizer, batch_size=32, shuffle=True):
    dataset = CodeSummarizationDataset(df, code_vocab, summary_vocab, code_tokenizer, summary_tokenizer)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x, code_vocab.stoi["<pad>"], summary_vocab.stoi["<pad>"])
    )
    return loader
