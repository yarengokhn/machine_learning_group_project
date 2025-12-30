import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch size, hid dim]
        # encoder_outputs: [src len, batch size, hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch size, src len, hid dim]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs: [batch size, src len, hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch size, src len, hid dim]
        
        attention = self.v(energy).squeeze(2)
        # attention: [batch size, src len]
        
        return F.softmax(attention, dim=1)
