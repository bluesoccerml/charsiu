import os
import math
import random
import concurrent.futures
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Config
from g2p_en import G2p
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 0. CONFIGURATION & SEED
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    sample_rate = 16000
    base_model_name = "facebook/wav2vec2-base"
    
    # Projection & Hidden Sizes
    embed_dim = 256
    proj_dim = 256
    num_negatives = 50
    temperature = 0.1
    lambda_fs = 1.0  # Weight for Forward-Sum loss
    
    # SpecAugment / Masking
    mask_time_prob_range = (0.05, 0.40)
    mask_time_length = 10
    
    # Training Parameters
    batch_size = 8
    grad_accum_steps = 4  # Effective batch size = 32
    learning_rate = 1e-5
    weight_decay = 1e-4
    epochs_per_curriculum = 2
    
    # Dataset Base Directory
    base_dir = "/kaggle/input/datasets/mozillaorg/common-voice"
    output_dir = "./charsiu_alignment_checkpoints"
    
    train_partitions = ["cv-other-train", "cv-valid-train"]
    val_partitions = ["cv-valid-dev", "cv-other-dev"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.seed)
os.makedirs(Config.output_dir, exist_ok=True)

# ==========================================
# 1. PHONE VOCABULARY & G2P PROCESSOR
# ==========================================
CMU_PHONEMES = [
    "[PAD]", "[UNK]", "[SIL]",
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "DX"
]

phone2id = {p: i for i, p in enumerate(CMU_PHONEMES)}
id2phone = {i: p for i, p in enumerate(CMU_PHONEMES)}
g2p = G2p()

def text_to_phone_ids(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return [phone2id["[SIL]"]]
    raw_phones = g2p(text)
    phone_ids = [phone2id["[SIL]"]]
    for p in raw_phones:
        cleaned = "".join([c for c in p if not c.isdigit()]).strip().upper()
        if cleaned in phone2id:
            phone_ids.append(phone2id[cleaned])
        elif cleaned in [" ", ""]:
            phone_ids.append(phone2id["[SIL]"])
        else:
            phone_ids.append(phone2id["[UNK]"])
    phone_ids.append(phone2id["[SIL]"])
    return phone_ids

# ==========================================
# 2. ROBUST DURATION & PATH RESOLUTION
# ==========================================
def get_audio_duration_robust(path):
    try:
        info = sf.info(path)
        if info.duration and info.duration > 0.05:
            return info.duration
    except Exception:
        pass

    try:
        info = torchaudio.info(path)
        if info.num_frames > 0 and info.sample_rate > 0:
            dur = info.num_frames / float(info.sample_rate)
            if dur > 0.05:
                return dur
    except Exception:
        pass

    try:
        wav, sr = torchaudio.load(path)
        dur = wav.shape[-1] / float(sr)
        return dur
    except Exception:
        return None

def resolve_audio_path(base_dir, partition, raw_filename):
    fname = os.path.basename(str(raw_filename))
    candidates = [
        os.path.join(base_dir, partition, partition, fname),
        os.path.join(base_dir, partition, fname),
        os.path.join(base_dir, partition, str(raw_filename)),
        os.path.join(base_dir, str(raw_filename))
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

def build_dataset_metadata(base_dir, partitions):
    all_rows = []
    print(f"Loading partitions: {partitions}")
    for part in partitions:
        csv_path = os.path.join(base_dir, f"{part}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if 'accent' in df.columns and 'text' in df.columns:
            df_accents = df[df['accent'].notna() & (df['accent'] != '')].copy()
            for _, row in df_accents.iterrows():
                resolved_path = resolve_audio_path(base_dir, part, row['filename'])
                if resolved_path is not None:
                    all_rows.append({
                        'absolute_path': resolved_path,
                        'text': row['text'],
                        'accent': row['accent']
                    })
    return pd.DataFrame(all_rows)

def _validate_single_sample(row_dict_and_limits):
    row_dict, min_dur, max_dur = row_dict_and_limits
    path = row_dict['absolute_path']
    dur = get_audio_duration_robust(path)
    if dur is not None and (min_dur <= dur <= max_dur):
        return {
            'path': path,
            'text': str(row_dict['text']),
            'duration': dur
        }
    return None

class CommonVoiceAlignmentDataset(Dataset):
    def __init__(self, df, max_duration=10.0, min_duration=0.5, max_workers=8):
        self.samples = []
        rows = [row.to_dict() for _, row in df.iterrows()]
        task_args = [(r, min_duration, max_duration) for r in rows]

        print(f"Validating {len(rows)} audio files (range: [{min_duration}s - {max_duration}s])...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(_validate_single_sample, task_args), total=len(task_args), desc="Processing"))

        self.samples = [r for r in results if r is not None]
        print(f"Retained {len(self.samples)} valid audio samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        waveform, sr = torchaudio.load(item['path'])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != Config.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=Config.sample_rate)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0)
        phone_ids = torch.tensor(text_to_phone_ids(item['text']), dtype=torch.long)
        return waveform, phone_ids, item['duration']

def alignment_collate_fn(batch):
    waveforms, phone_seqs, durations = zip(*batch)
    audio_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    padded_audio = nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    phone_lengths = torch.tensor([len(p) for p in phone_seqs], dtype=torch.long)
    padded_phones = nn.utils.rnn.pad_sequence(phone_seqs, batch_first=True, padding_value=phone2id["[PAD]"])
    return padded_audio, audio_lengths, padded_phones, phone_lengths

# ==========================================
# 3. FORWARD-SUM MONOTONIC LOSS[cite: 2]
# ==========================================
def forward_sum_loss(log_attn_matrix, text_lens, audio_lens):
    B, N, T = log_attn_matrix.shape
    device = log_attn_matrix.device
    dtype = log_attn_matrix.dtype

    alpha = torch.full((B, N), -1e9, device=device, dtype=dtype)
    alpha[:, 0] = log_attn_matrix[:, 0, 0]
    
    alphas = [alpha]
    for t in range(1, T):
        prev_stay = alpha
        prev_trans = torch.cat([torch.full((B, 1), -1e9, device=device, dtype=dtype), alpha[:, :-1]], dim=1)
        log_trans = torch.logaddexp(prev_stay, prev_trans)
        alpha = log_trans + log_attn_matrix[:, :, t]
        alphas.append(alpha)
        
    alphas = torch.stack(alphas, dim=2)
    
    loss = 0.0
    for b in range(B):
        n_idx = text_lens[b] - 1
        t_idx = audio_lens[b] - 1
        log_prob = alphas[b, n_idx, t_idx]
        loss += -log_prob / audio_lens[b].float().clamp(min=1.0)
        
    return loss / B

# ==========================================
# 4. CHARSIU W2V2-FS ARCHITECTURE[cite: 2]
# ==========================================
class PhoneEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=phone2id["[PAD]"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, phone_ids, phone_lengths):
        mask = (phone_ids == phone2id["[PAD]"])
        x = self.embedding(phone_ids)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.transpose(1, 2)
        x = F.gelu(self.conv(x)).transpose(1, 2)
        return x

class CharsiuW2V2FS(nn.Module):
    def __init__(self, cfg=Config):
        super().__init__()
        self.cfg = cfg
        
        pretrain_model = Wav2Vec2ForPreTraining.from_pretrained(cfg.base_model_name)
        self.w2v2 = pretrain_model.wav2vec2
        self.quantizer = pretrain_model.quantizer
        self.project_q = pretrain_model.project_q
        
        self.quantizer.requires_grad_(False)
        self.project_q.requires_grad_(False)
        
        speech_dim = self.w2v2.config.hidden_size
        quant_dim = pretrain_model.config.proj_codevector_dim
        
        self.phone_encoder = PhoneEncoder(vocab_size=len(CMU_PHONEMES), embed_dim=cfg.embed_dim)
        self.fx = nn.Linear(speech_dim, cfg.proj_dim)
        self.fy = nn.Linear(cfg.embed_dim, cfg.proj_dim)
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(speech_dim + cfg.embed_dim, speech_dim),
            nn.GELU(),
            nn.Linear(speech_dim, quant_dim)
        )

    def apply_time_masking(self, hidden_states):
        B, T, D = hidden_states.shape
        p = random.uniform(*self.cfg.mask_time_prob_range)
        num_mask_spans = int(p * T / self.cfg.mask_time_length)
        
        masked_states = hidden_states.clone()
        for b in range(B):
            for _ in range(num_mask_spans):
                start = random.randint(0, max(0, T - self.cfg.mask_time_length))
                masked_states[b, start:start + self.cfg.mask_time_length, :] = 0.0
        return masked_states

    def sample_negatives(self, quant_features, num_negatives=50):
        B, T, D = quant_features.shape
        quant_flat = quant_features.reshape(B * T, D)
        indices = torch.randint(0, B * T, (B, T, num_negatives), device=quant_features.device)
        return quant_flat[indices] 

    def forward(self, audio, audio_lens, phone_ids, phone_lens):
        # 1. Extract raw CNN features
        extract_features = self.w2v2.feature_extractor(audio)
        extract_features = extract_features.transpose(1, 2) # (B, T, 512)
        
        # 2. Get Quantized Vectors using raw 512-dim features
        with torch.no_grad():
            quant_out = self.quantizer(extract_features)
            quant_feats = quant_out[0] if isinstance(quant_out, tuple) else quant_out
            quant_proj = self.project_q(quant_feats) # (B, T, D_q)
        
        # 3. Project features to 768-dim for the Transformer Encoder
        hidden_states, _ = self.w2v2.feature_projection(extract_features)
        
        masked_features = self.apply_time_masking(hidden_states) if self.training else hidden_states
        encoder_out = self.w2v2.encoder(masked_features)
        speech_hidden = encoder_out.last_hidden_state # (B, T, 768)
        
        B, T, _ = speech_hidden.shape
        effective_audio_lens = torch.clamp((audio_lens // 320), min=1, max=T)
        
        phone_hidden = self.phone_encoder(phone_ids, phone_lens) 
        
        fx_proj = F.normalize(self.fx(speech_hidden), dim=-1)
        fy_proj = F.normalize(self.fy(phone_hidden), dim=-1)
        
        D = torch.bmm(fy_proj, fx_proj.transpose(1, 2)) / math.sqrt(self.cfg.proj_dim)
        A = F.softmax(D, dim=1) 
        log_A = F.log_softmax(D, dim=1)
        
        Y_aligned = torch.bmm(A.transpose(1, 2), phone_hidden)
        H = torch.cat([speech_hidden, Y_aligned], dim=-1)
        H_proj = F.normalize(self.fusion_proj(H), dim=-1)
        
        # 4. Contrastive Loss (Lm)[cite: 2]
        pos_sim = (H_proj * F.normalize(quant_proj, dim=-1)).sum(dim=-1, keepdim=True) / self.cfg.temperature
        negatives = self.sample_negatives(F.normalize(quant_proj, dim=-1), self.cfg.num_negatives)
        H_proj_expanded = H_proj.unsqueeze(2)
        neg_sim = (H_proj_expanded * negatives).sum(dim=-1) / self.cfg.temperature
        
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        targets = torch.zeros(B * T, dtype=torch.long, device=audio.device)
        
        loss_m = F.cross_entropy(logits.reshape(B * T, -1), targets)
        
        # 5. Forward-Sum Loss (Lfs)[cite: 2]
        loss_fs = forward_sum_loss(log_A, phone_lens, effective_audio_lens)
        total_loss = loss_m + self.cfg.lambda_fs * loss_fs
        
        return {
            "loss": total_loss,
            "loss_m": loss_m,
            "loss_fs": loss_fs,
            "attention": A
        }

# ==========================================
# 5. HEATMAP PLOTTER & TRAINING PIPELINE
# ==========================================
def plot_attention_matrix(attention_matrix, phones, epoch, stage_name):
    plt.figure(figsize=(12, 6))
    plt.imshow(attention_matrix, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    plt.yticks(range(len(phones)), phones, fontsize=8)
    plt.xlabel('Acoustic Frames (20ms step)')
    plt.ylabel('Phonemes')
    plt.title(f'Alignment Attention Heatmap | {stage_name} (Epoch {epoch})')
    plt.colorbar(label='Probability')
    plt.tight_layout()
    plt.show()
    plt.close()

def evaluate_and_plot(model, val_loader, cfg, epoch, stage_name):
    model.eval()
    print("\n[Evaluation] Extracting validation sample and plotting alignment heatmap...")
    with torch.no_grad():
        for audio, audio_lens, phones, phone_lens in val_loader:
            audio = audio.to(cfg.device)
            audio_lens = audio_lens.to(cfg.device)
            phones = phones.to(cfg.device)
            phone_lens = phone_lens.to(cfg.device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(audio, audio_lens, phones, phone_lens)
                
            attn = outputs["attention"]
            actual_T = (audio_lens[0] // 320).item()
            actual_N = phone_lens[0].item()
            
            attn_0 = attn[0, :actual_N, :actual_T].cpu().numpy()
            phone_ids_0 = phones[0, :actual_N].cpu().tolist()
            phone_labels = [id2phone[p] for p in phone_ids_0]
            
            plot_attention_matrix(attn_0, phone_labels, epoch, stage_name)
            break
    model.train()

def train_one_epoch(model, dataloader, optimizer, scaler, cfg, epoch):
    model.train()
    total_loss_accum, total_m_accum, total_fs_accum = 0.0, 0.0, 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for step, (audio, audio_lens, phones, phone_lens) in enumerate(pbar):
        audio = audio.to(cfg.device)
        audio_lens = audio_lens.to(cfg.device)
        phones = phones.to(cfg.device)
        phone_lens = phone_lens.to(cfg.device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(audio, audio_lens, phones, phone_lens)
            loss = outputs["loss"] / cfg.grad_accum_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss_accum += outputs["loss"].item()
        total_m_accum += outputs["loss_m"].item()
        total_fs_accum += outputs["loss_fs"].item()
        
        pbar.set_postfix({
            "Total": f"{total_loss_accum / (step + 1):.3f}",
            "Lm(Contr)": f"{total_m_accum / (step + 1):.3f}",
            "Lfs(Mono)": f"{total_fs_accum / (step + 1):.3f}"
        })
        
    print(f"\n=> Epoch {epoch} Averages: Total Loss: {total_loss_accum/len(dataloader):.4f} | "
          f"Contrastive (Lm): {total_m_accum/len(dataloader):.4f} | "
          f"Forward-Sum (Lfs): {total_fs_accum/len(dataloader):.4f}")

def run_curriculum_training():
    print("Building Training Metadata...")
    train_df = build_dataset_metadata(Config.base_dir, Config.train_partitions)
    
    print("Building Validation Metadata...")
    val_df = build_dataset_metadata(Config.base_dir, Config.val_partitions)
    
    if train_df.empty:
        print("Error: Train dataframe empty. Check folder structure.")
        return

    if val_df.empty or len(val_df) < 10:
        print("Validation dataframe is small/empty. Taking a 500-sample holdout from training data...")
        val_df = train_df.sample(n=min(500, len(train_df)), random_state=Config.seed)
        train_df = train_df.drop(val_df.index)

    print(f"Total Train Metadata: {len(train_df)} | Total Val Metadata: {len(val_df)}")

    val_dataset = CommonVoiceAlignmentDataset(val_df, min_duration=0.5, max_duration=10.0)
    if len(val_dataset) == 0:
        print("Fallback: Using direct train samples for validation to guarantee eval.")
        val_dataset = CommonVoiceAlignmentDataset(train_df.head(200), min_duration=0.1, max_duration=15.0)

    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=True, 
                            collate_fn=alignment_collate_fn, num_workers=2)

    # Curriculum Stages[cite: 2]
    curriculum_stages = [
        {"name": "Chunk_1 (<3s)", "min": 0.3, "max": 3.0},
        {"name": "Chunk_2 (3s-5s)", "min": 3.0, "max": 5.0},
        {"name": "Chunk_3 (5s-10s)", "min": 5.0, "max": 10.0},
    ]
    
    print("Initializing Charsiu W2V2-FS Model...")
    model = CharsiuW2V2FS(Config).to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    
    scaler = torch.amp.GradScaler('cuda')

    for stage_idx, stage in enumerate(curriculum_stages):
        print(f"\n==========================================")
        print(f" Starting Curriculum Stage {stage_idx+1}: {stage['name']}")
        print(f"==========================================")
        
        stage_dataset = CommonVoiceAlignmentDataset(
            train_df, min_duration=stage["min"], max_duration=stage["max"]
        )
        
        if len(stage_dataset) == 0:
            print(f"No samples found for {stage['name']}. Skipping...")
            continue
            
        stage_loader = DataLoader(
            stage_dataset, batch_size=Config.batch_size, shuffle=True,
            collate_fn=alignment_collate_fn, num_workers=2, pin_memory=True
        )
        
        for epoch in range(1, Config.epochs_per_curriculum + 1):
            train_one_epoch(model, stage_loader, optimizer, scaler, Config, epoch)
            evaluate_and_plot(model, val_loader, Config, epoch, stage["name"])
            
        save_path = os.path.join(Config.output_dir, f"charsiu_stage_{stage_idx+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved stage checkpoint to {save_path}")

# Run training
if __name__ == "__main__":
    run_curriculum_training()

