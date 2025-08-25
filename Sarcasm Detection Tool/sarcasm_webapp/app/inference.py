import os
import numpy as np
import torch
import librosa, soundfile as sf
from torch_geometric.data import Data

from .model_def import CMGAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy-load HF models so the server can start even if downloads take time.
_text_tokenizer = _text_model = _wav_feat_proc = _wav_feat_model = _asr_proc = _asr_model = None

def _ensure_models_loaded():
    """Load Hugging Face models only once (on first use)."""
    global _text_tokenizer, _text_model, _wav_feat_proc, _wav_feat_model, _asr_proc, _asr_model
    if _text_model is not None:
        return
    from transformers import (
        RobertaTokenizer, RobertaModel,
        Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
    )
    _text_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    _text_model = RobertaModel.from_pretrained("roberta-base").to(DEVICE)
    _wav_feat_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    _wav_feat_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    _asr_proc = _wav_feat_proc
    _asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)

def to_wav_16k_mono(src_path: str, dst_path: str, target_sr: int = 16000):
    y, sr = librosa.load(src_path, sr=target_sr, mono=True)
    sf.write(dst_path, y, target_sr)
    return dst_path

def transcribe_text_from_wav(wav_path: str) -> str:
    _ensure_models_loaded()
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    inputs = _asr_proc(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        logits = _asr_model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = _asr_proc.batch_decode(pred_ids)[0]
    return text

def extract_text_features(text: str) -> torch.Tensor:
    _ensure_models_loaded()
    inputs = _text_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = _text_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).detach().cpu()  # (T_text, 768)

def extract_audio_features(wav_path: str) -> torch.Tensor:
    _ensure_models_loaded()
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    inputs = _wav_feat_proc(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = _wav_feat_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).detach().cpu()  # (T_audio, 768)

def extract_prosody_features(wav_path: str) -> torch.Tensor:
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if len(y) < 320:
        y = np.pad(y, (0, 320 - len(y)))
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=1024, hop_length=256)
    pitch_mean = float(np.nanmean(f0))
    rms = float(librosa.feature.rms(y=y, frame_length=1024, hop_length=256).mean())
    return torch.tensor([[pitch_mean, rms]], dtype=torch.float32)  # (1,2)

def build_graph(text_feat: torch.Tensor, audio_feat: torch.Tensor, prosody_feat: torch.Tensor) -> Data:
    # Pad prosody to 768 dims
    pad = torch.nn.functional.pad(prosody_feat, (0, 766))  # (1,768)
    x = torch.cat([text_feat, audio_feat, pad], dim=0)

    # Node types: 0=text, 1=audio, 2=prosody, 3=global
    node_type = torch.cat([
        torch.zeros(text_feat.size(0), dtype=torch.long),
        torch.ones(audio_feat.size(0), dtype=torch.long),
        torch.full((1,), 2, dtype=torch.long),
    ])
    # Add global node (zeros)
    global_node = torch.zeros(1, 768)
    x = torch.cat([x, global_node], dim=0)
    node_type = torch.cat([node_type, torch.full((1,), 3, dtype=torch.long)], dim=0)

    # Fully connected undirected edges
    num_nodes = x.size(0)
    idx = torch.arange(num_nodes)
    pairs = torch.combinations(idx, r=2).t()  # (2, E)
    edge_index = torch.cat([pairs, pairs.flip(0)], dim=1)
    return Data(x=x, edge_index=edge_index, node_type=node_type)

def load_model(weights_path: str, hidden_dim: int, num_layers: int) -> CMGAT:
    model = CMGAT(in_dim=768, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers).to(DEVICE)
    sd = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

def predict_from_file(audio_path: str, weights_path: str, hidden_dim: int, num_layers: int):
    """Main API used by FastAPI; DO NOT rename."""
    # 1) Convert to WAV 16k mono
    tmp_wav = os.path.splitext(audio_path)[0] + "_16k.wav"
    to_wav_16k_mono(audio_path, tmp_wav)

    # 2) Transcribe text
    transcript = transcribe_text_from_wav(tmp_wav)

    # 3) Extract features
    text_feat = extract_text_features(transcript)
    audio_feat = extract_audio_features(tmp_wav)
    prosody_feat = extract_prosody_features(tmp_wav)  # (1,2)

    # 4) Build graph
    data = build_graph(text_feat, audio_feat, prosody_feat).to(DEVICE)

    # 5) Load model and predict
    model = load_model(weights_path, hidden_dim=hidden_dim, num_layers=num_layers)
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy().tolist()
        pred_idx = int(np.argmax(probs))
    label_map = {0: "Not Sarcastic", 1: "Sarcastic"}
    return {
        "prediction": label_map.get(pred_idx, str(pred_idx)),
        "probabilities": {"Not Sarcastic": float(probs[0]), "Sarcastic": float(probs[1])},
        "transcript": transcript
    }
