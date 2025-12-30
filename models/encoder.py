import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch size, src len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch size, src len, emb dim]
        
        # Transpose for RNN: [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded.transpose(0, 1))
        # outputs: [src len, batch size, hid dim * 2]
        # hidden: [n layers * 2, batch size, hid dim]
        
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are [forward_hi, backward_hi]
        
        # Initial hidden state for decoder is the final forward and backward encoder hidden states
        # concat and passed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        return outputs, hidden
