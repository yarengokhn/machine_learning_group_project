import torch
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

def calculate_metrics(hypotheses, references):
    """
    hypotheses: list of strings
    references: list of strings (ground truth)
    """
    # BLEU
    # corpus_bleu expects list of list of tokens for references and list of tokens for hypotheses
    ref_tokens = [[ref.split()] for ref in references]
    hyp_tokens = [hyp.split() for hyp in hypotheses]
    bleu = corpus_bleu(ref_tokens, hyp_tokens)
    
    # ROUGE
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    
    return {
        "bleu": bleu,
        "rouge": scores
    }

def get_text_from_ids(ids, vocab):
    tokens = vocab.decode(ids.tolist())
    # Remove special tokens
    tokens = [t for t in tokens if t not in ["<pad>", "<sos>", "<eos>"]]
    return " ".join(tokens)
