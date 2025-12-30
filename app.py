from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import os
import sys
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_utils import Tokenizer, load_vocab
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from src.inference import InferenceEngine

app = Flask(__name__, static_folder='static')
CORS(app)

# Global variables for model and engine
engine = None

def init_engine():
    global engine
    print("Initializing Inference Engine...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config_path = "configs/base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    emb_dim = config.get('embedding_dim', 256)
    hid_dim = config.get('hidden_dim', 512)
    n_layers = config.get('n_layers', 1)
    dropout = config.get('dropout', 0.5)
    
    # Load Vocabs
    try:
        code_vocab = load_vocab("checkpoints/improved_v1_code_vocab.pkl")
        summary_vocab = load_vocab("checkpoints/improved_v1_summary_vocab.pkl")
    except FileNotFoundError:
        print("Vocab files not found. Inference will fail.")
        return

    code_tokenizer = Tokenizer(is_code=True)
    
    # Init Model
    print("Loading vocabularies...")
    attn = Attention(hid_dim)
    enc = Encoder(len(code_vocab), emb_dim, hid_dim, n_layers, dropout)
    dec = Decoder(len(summary_vocab), emb_dim, hid_dim, n_layers, dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # Load Weights
    print("Loading model weights...")
    checkpoint_path = "checkpoints/improved_v1.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model weights loaded.")
    else:
        print("Model weights not found. Inference will produce untrained results.")

    engine = InferenceEngine(model, code_vocab, summary_vocab, code_tokenizer, device)
    print("Inference Engine ready.")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/summarize', methods=['POST'])
def summarize():
    if engine is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    data = request.json
    code = data.get('code')
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    try:
        summary = engine.summarize(code)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_engine()
    app.run(host='0.0.0.0', port=5003, debug=False)
