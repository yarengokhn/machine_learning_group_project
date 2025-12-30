import torch

class InferenceEngine:
    def __init__(self, model, code_vocab, summary_vocab, code_tokenizer, device):
        self.model = model
        self.code_vocab = code_vocab
        self.summary_vocab = summary_vocab
        self.code_tokenizer = code_tokenizer
        self.device = device

    def summarize(self, code_snippet, max_len=30, beam_size=5):
        self.model.eval()
        
        # Tokenize and numericalize
        tokens = self.code_tokenizer.tokenize(code_snippet)
        src_ids = self.code_vocab.encode(tokens)
        src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)

        # Beam search: (score, trg_ids, hidden_state)
        # score is the log probability of the sequence
        beams = [(0.0, [self.summary_vocab.stoi["<sos>"]], hidden)]
        
        for i in range(max_len):
            all_candidates = []
            
            for score, trg_ids, current_hidden in beams:
                # If sequence already finished, keep it as a candidate
                if trg_ids[-1] == self.summary_vocab.stoi["<eos>"]:
                    all_candidates.append((score, trg_ids, current_hidden))
                    continue
                
                trg_tensor = torch.LongTensor([trg_ids[-1]]).to(self.device)
                
                with torch.no_grad():
                    output, next_hidden = self.model.decoder(trg_tensor, current_hidden, encoder_outputs)
                
                # Apply log_softmax to get scores
                log_probs = torch.log_softmax(output, dim=1).squeeze(0)
                
                # --- Repetition Penalty ---
                # Penalize tokens that have appeared recently in the sequence
                for token_id in set(trg_ids[-5:]): # Look at last 5 tokens
                    if token_id not in [self.summary_vocab.stoi["<pad>"], self.summary_vocab.stoi["<unk>"]]:
                        log_probs[token_id] -= 5.0 # Strong penalty to prevent loops
                
                # Get top-k transitions
                top_probs, top_ids = log_probs.topk(beam_size)
                
                for k in range(beam_size):
                    candidate_score = score + top_probs[k].item()
                    candidate_ids = trg_ids + [top_ids[k].item()]
                    all_candidates.append((candidate_score, candidate_ids, next_hidden))
            
            # Select top-k beams for the next step
            # Sort by score (higher is better)
            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            beams = ordered[:beam_size]
            
            # If all top beams have finished, we can stop early
            if all(b[1][-1] == self.summary_vocab.stoi["<eos>"] for b in beams):
                break
        
        # Select the best beam
        best_trg_ids = beams[0][1]
        
        # Decode
        summary_tokens = self.summary_vocab.decode(best_trg_ids)
        # Filter out special tokens
        summary = " ".join([t for t in summary_tokens if t not in ["<sos>", "<eos>", "<pad>"]])
        
        return summary
