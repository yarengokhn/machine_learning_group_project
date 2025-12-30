import torch
import os
import argparse
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import Tokenizer, load_vocab
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from src.inference import InferenceEngine

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Vocabs
    try:
        code_vocab = load_vocab("checkpoints/code_vocab.pkl")
        summary_vocab = load_vocab("checkpoints/summary_vocab.pkl")
    except FileNotFoundError:
        print("Vocab files not found. Please train the model first.")
        return

    code_tokenizer = Tokenizer(is_code=True)
    
    # Init Model
    attn = Attention(args.hid_dim)
    enc = Encoder(len(code_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout)
    dec = Decoder(len(summary_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # Load Weights
    if os.path.exists("checkpoints/best_model.pt"):
        model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    else:
        print("Model weights not found. Please train the model first.")
        return

    engine = InferenceEngine(model, code_vocab, summary_vocab, code_tokenizer, device)
    
    if args.input:
        summary = engine.summarize(args.input)
        print(f"\nGenerated Summary: {summary}")
    else:
        print("Please provide a python function using --input")

if __name__ == "__main__":
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--input", type=str, help="Python function to summarize")
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'embedding_dim' in config: args.emb_dim = config['embedding_dim']
            if 'hidden_dim' in config: args.hid_dim = config['hidden_dim']
            if 'n_layers' in config: args.n_layers = config['n_layers']
            if 'dropout' in config: args.dropout = config['dropout']
    
    main(args)
