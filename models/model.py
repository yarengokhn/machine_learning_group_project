import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM Katmanı:
        # n_layers=2 yaparak modelin daha derin özellikleri öğrenmesini sağlıyoruz.
        # dropout ekleyerek modelin ezberlemesini (overfitting) önlüyoruz.
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            batch_first=True, 
                            dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, sequence_length] (Token ID'leri)
        
        embedded = self.dropout(self.embedding(x))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        # outputs: her adım için gizli durumlar
        # (hidden, cell): son adımın gizli durumu ve hücre hafızası
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, hidden, cell
    

class HybridEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, n_heads=8, dropout=0.3):
        super().__init__()
        
        # 1. Aşama: Sayıları Vektöre Çevirme
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Aşama: LSTM (Dizisel yapıları anlamak için)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # LSTM bidirectional olduğu için hidden_dim * 2 olur.
        # Bunu Transformer'ın beklediği boyuta indirelim.
        self.reduce_dim = nn.Linear(hidden_dim * 2, hidden_dim)

        # 3. Aşama: Transformer Layer (Global bağlamı anlamak için)
        # Sadece Encoder katmanını kullanıyoruz.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.dropout(self.embedding(x))
        
        # Önce LSTM'den geçirelim
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Boyutu Transformer için ayarla
        lstm_out = self.reduce_dim(lstm_out)
        
        # Sonra Transformer'dan geçirelim (Daha güçlü dikkat/attention sağlar)
        transformer_out = self.transformer_encoder(lstm_out)
        
        # hidden ve cell'i decoder'a göndermek üzere hazırlayalım 
        # (Bidirectional olduğu için katmanları birleştiriyoruz)
        hidden = torch.mean(hidden, dim=0, keepdim=True) # Basitleştirilmiş birleştirme
        cell = torch.mean(cell, dim=0, keepdim=True)
        
        return transformer_out, hidden, cell
    
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Decoder LSTM'i
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            batch_first=True, dropout=dropout)
        
        # En son katman: LSTM çıktısını kelime tahminine (olasılıklara) dönüştürür
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, cell):
        # input_step: O an üretilen/verilen tek bir kelime [batch_size, 1]
        # hidden, cell: Encoder'dan gelen veya bir önceki adımdan aktarılan hafıza
        
        input_step = input_step.unsqueeze(1) # [batch_size, 1] formatına getir
        embedded = self.dropout(self.embedding(input_step))
        
        # LSTM adımını çalıştır
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Tahmin yap: Bir sonraki kelime hangisi olabilir?
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell    
    

class HybridDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Decoder LSTM'i (Encoder ile aynı hidden_dim boyutunda olmalı)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            batch_first=True, dropout=dropout)
        
        # Kelime tahmini yapan katman
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, cell):
        # input_step: O anki kelime ID'si [batch_size]
        # hidden, cell: Encoder'dan gelen özetlenmiş kod bilgisi
        
        # [batch_size] -> [batch_size, 1] (LSTM tek seferde bir adım işler)
        input_step = input_step.unsqueeze(1)
        
        embedded = self.dropout(self.embedding(input_step))
        
        # LSTM adımı: "Şu anki kelimeyi ve kod bilgisini birleştir, sonrakini düşün"
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Çıktıyı kelime olasılıklarına çevir
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell    
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source: Tokenize edilmiş Python kodu
        # target: Tokenize edilmiş özet (eğitim için)
        
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.vocab_size
        
        # Tahminleri saklamak için boş bir kutu
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # 1. Kod Encoder'dan geçer: Transformer ve LSTM bilgisi üretilir
        transformer_out, hidden, cell = self.encoder(source)
        
        # 2. Decoder'ın ilk girişi: <SOS> (Start of Sentence) tokenı
        input_step = target[:, 0]
        
        for t in range(1, target_len):
            # Decoder'a o anki kelimeyi ve Encoder'dan gelen kod hafızasını veriyoruz
            output, hidden, cell = self.decoder(input_step, hidden, cell)
            
            # Tahmini kaydet
            outputs[:, t] = output
            
            # Bir sonraki adımda neyi girdi olarak vereceğiz?
            top1 = output.argmax(1) # Modelin en yüksek ihtimal verdiği kelime
            
            # Teacher Forcing: Eğitimde bazen doğru kelimeyi, bazen modelin tahminini kullanırız
            import random
            input_step = target[:, t] if random.random() < teacher_forcing_ratio else top1
            
        return outputs    