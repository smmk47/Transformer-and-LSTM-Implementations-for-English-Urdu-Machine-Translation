# app.py

import os
# Set environment variable to handle OpenMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import sentencepiece as spm
import re
import math
import logging
from collections import defaultdict

# Define device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Flask app
app = Flask(__name__)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

#############################################
# Load SentencePiece Models
#############################################

# Initialize SentencePiece processors
sp_en = spm.SentencePieceProcessor()
sp_en.load('spm_en.model')

sp_ur = spm.SentencePieceProcessor()
sp_ur.load('spm_ur.model')

# Verify vocabulary sizes
en_vocab_size = sp_en.get_piece_size()
ur_vocab_size = sp_ur.get_piece_size()
logging.info(f"English Vocabulary Size: {en_vocab_size}")  # Should be 7256
logging.info(f"Urdu Vocabulary Size: {ur_vocab_size}")      # Should be 6735

# Define vocabulary sizes based on training
SRC_VOCAB_SIZE = 8000  # English
TGT_VOCAB_SIZE = 8000  # Urdu

# Check if SentencePiece vocab sizes match training
if en_vocab_size != SRC_VOCAB_SIZE or ur_vocab_size != TGT_VOCAB_SIZE:
    logging.error("SentencePiece vocabulary sizes do not match training configuration.")
    logging.error(f"Expected English Vocab Size: {SRC_VOCAB_SIZE}, Found: {en_vocab_size}")
    logging.error(f"Expected Urdu Vocab Size: {TGT_VOCAB_SIZE}, Found: {ur_vocab_size}")
    raise ValueError("Vocabulary size mismatch. Please ensure SentencePiece models match training vocab sizes.")

#############################################
# Define Models
#############################################

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Odd indices (if d_model is odd)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)      # Odd indices
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Not a parameter, but persistent

    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# Transformer Model Class
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        # Embedding layers
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                         num_encoder_layers=num_encoder_layers,
                                         num_decoder_layers=num_decoder_layers,
                                         dim_feedforward=dim_feedforward, dropout=dropout,
                                         batch_first=True)  # Set batch_first=True

        # Output generator
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        # Embed source and target
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        
        # Pass through Transformer encoder and decoder
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)
        output = self.generator(out)  # Shape: [batch_size, tgt_len, tgt_vocab_size]
        return output

# LSTM Seq2Seq Model Class
class Seq2SeqLSTM(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.1):
        super(Seq2SeqLSTM, self).__init__()
        # Embedding layers
        self.embedding_src = nn.Embedding(src_vocab_size, embed_size)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, embed_size)
        
        # Encoder and Decoder LSTM
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        # Encode source
        embedded_src = self.dropout(self.embedding_src(src))
        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)
        
        # Decode target
        embedded_tgt = self.dropout(self.embedding_tgt(tgt))
        decoder_outputs, _ = self.decoder(embedded_tgt, (hidden, cell))
        
        # Generate output
        outputs = self.fc_out(decoder_outputs)  # Shape: [batch_size, tgt_len, tgt_vocab_size]
        return outputs

#############################################
# Initialize Models
#############################################

# Initialize Transformer Model
transformer_model = TransformerModel(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=512,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1
).to(device)

# Initialize LSTM Model
lstm_model = Seq2SeqLSTM(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    embed_size=256,
    hidden_size=512,
    num_layers=2,
    dropout=0.1
).to(device)

# Load trained Transformer weights
try:
    transformer_state_dict = torch.load('best_transformer_model.pth', map_location=device)
    transformer_model.load_state_dict(transformer_state_dict)
    transformer_model.eval()
    logging.info("Transformer model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Transformer model: {e}")
    raise e

# Load trained LSTM weights
try:
    lstm_state_dict = torch.load('best_lstm_model.pth', map_location=device)
    lstm_model.load_state_dict(lstm_state_dict)
    lstm_model.eval()
    logging.info("LSTM model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading LSTM model: {e}")
    raise e

#############################################
# Preprocessing Function
#############################################

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # Remove special characters and digits
    sentence = re.sub(r"[^a-zA-Zا-ے۰-۹\s]+", "", sentence)
    # Remove extra spaces
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence

#############################################
# Translation Functions
#############################################

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

def translate_sentence_transformer(model, src_sentence):
    model.eval()
    src_sentence = preprocess_sentence(src_sentence)
    # Encode using SentencePiece
    src_indices = sp_en.encode(src_sentence, out_type=int)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)  # [1, src_len]
    num_tokens = src_tensor.size(1)
    src_mask = torch.zeros((num_tokens, num_tokens)).type(torch.bool).to(device)
    
    with torch.no_grad():
        src_emb = model.positional_encoding(model.src_tok_emb(src_tensor) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=(src_tensor == sp_en.pad_id()))
        ys = torch.ones(1, 1).fill_(sp_ur.bos_id()).type(torch.long).to(device)
        for i in range(100):  # max length
            tgt_emb = model.positional_encoding(model.tgt_tok_emb(ys) * math.sqrt(model.d_model))
            tgt_mask = generate_square_subsequent_mask(ys.size(1))
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=(ys == sp_ur.pad_id()),
                                           memory_key_padding_mask=(src_tensor == sp_en.pad_id()))
            out = model.generator(out)  # [1, tgt_len, vocab_size]
            prob = out[:, -1, :].softmax(dim=1)
            next_word = prob.argmax(dim=1).item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=1)
            if next_word == sp_ur.eos_id():
                break
    tgt_indices = ys[0, 1:].cpu().numpy()
    # Decode using SentencePiece
    tgt_tokens = sp_ur.decode(tgt_indices)
    return tgt_tokens.strip()

def translate_sentence_lstm(model, src_sentence):
    model.eval()
    src_sentence = preprocess_sentence(src_sentence)
    # Encode using SentencePiece
    src_indices = sp_en.encode(src_sentence, out_type=int)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)  # [1, src_len]
    
    with torch.no_grad():
        embedded_src = model.dropout(model.embedding_src(src_tensor))
        encoder_outputs, (hidden, cell) = model.encoder(embedded_src)
        
        ys = torch.ones(1, 1).fill_(sp_ur.bos_id()).type(torch.long).to(device)
        outputs = []
        for i in range(100):  # max length
            embedded_tgt = model.dropout(model.embedding_tgt(ys))
            decoder_outputs, (hidden, cell) = model.decoder(embedded_tgt, (hidden, cell))
            output = model.fc_out(decoder_outputs[:, -1, :])  # [1, vocab_size]
            prob = output.softmax(dim=1)
            next_word = prob.argmax(dim=1).item()
            if next_word == sp_ur.eos_id():
                break
            outputs.append(next_word)
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=1)
    # Decode using SentencePiece
    tgt_tokens = sp_ur.decode(outputs)
    return tgt_tokens.strip()

#############################################
# Flask Routes
#############################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text', '')
    model_type = data.get('model', 'transformer')  # 'transformer' or 'lstm'
    
    if not text.strip():
        return jsonify({'translation': 'Please enter some text to translate.'}), 400

    try:
        if model_type == 'transformer':
            translation = translate_sentence_transformer(transformer_model, text)
        elif model_type == 'lstm':
            translation = translate_sentence_lstm(lstm_model, text)
        else:
            return jsonify({'error': 'Invalid model type selected.'}), 400
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return jsonify({'error': 'An internal error occurred during translation.'}), 500
    
    return jsonify({'translation': translation})

if __name__ == '__main__':
    app.run(debug=True)
