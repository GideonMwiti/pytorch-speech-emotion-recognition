"""
Project 60: Speech Emotion Recognition (SER) with CRNN & MFCC Features
Classifies emotional tone from raw speech waveforms using 40-band MFCCs,
first-order deltas, second-order deltas, and a deep PyTorch CRNN architecture
(2D-CNN + Bidirectional LSTM + Self-Attention Temporal Pooling).
"""

import os
import sys
import time
import json
import numpy as np
import soundfile as sf
import scipy.signal
import scipy.fftpack
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Configure plot styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]
EMOTION2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2EMOTION = {i: e for i, e in enumerate(EMOTIONS)}


# =====================================================================
# 1. Acoustic Signal Processing & MFCC Feature Extraction Engine
# =====================================================================
class MFCCFeatureExtractor:
    """Computes 40-channel MFCCs, Delta, and Delta-Delta acoustic features."""

    def __init__(self, sr: int = 16000, n_mfcc: int = 40, n_mels: int = 40, 
                 n_fft: int = 512, hop_length: int = 160, target_frames: int = 128):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_frames = target_frames
        self.fbank = self._build_mel_filterbank()

    def _build_mel_filterbank(self) -> np.ndarray:
        """Constructs triangular Mel frequency filterbank matrix."""
        mel_min = 0.0
        mel_max = 2595.0 * np.log10(1.0 + (self.sr / 2.0) / 700.0)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)
        
        fbank = np.zeros((self.n_mels, int(np.floor(self.n_fft / 2 + 1))))
        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1] + 1e-8)
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m] + 1e-8)
        return fbank

    def _compute_delta(self, feat: np.ndarray, N: int = 2) -> np.ndarray:
        """Computes delta (velocity/acceleration) coefficients along time axis."""
        num_frames = feat.shape[1]
        denominator = 2 * sum(i ** 2 for i in range(1, N + 1))
        delta = np.zeros_like(feat)
        padded = np.pad(feat, ((0, 0), (N, N)), mode='edge')
        for t in range(num_frames):
            delta[:, t] = np.sum([n * (padded[:, t + N + n] - padded[:, t + N - n]) for n in range(1, N + 1)], axis=0) / denominator
        return delta

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts 3-channel feature tensor: [MFCC, Delta, DeltaDelta].
        Returns shape: (3, n_mfcc=40, target_frames=128)
        """
        # 1. Pre-emphasis filter
        y_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # 2. Short-Time Fourier Transform (STFT)
        _, _, Zxx = scipy.signal.stft(
            y_pre, 
            fs=self.sr, 
            nperseg=self.n_fft, 
            noverlap=self.n_fft - self.hop_length, 
            window='hamming'
        )
        power_spectrum = np.abs(Zxx) ** 2
        
        # 3. Mel filterbank application & log energy
        mel_energies = np.dot(self.fbank, power_spectrum)
        mel_energies = np.where(mel_energies <= 0, 1e-10, mel_energies)
        log_mel = 20.0 * np.log10(mel_energies)
        
        # 4. Discrete Cosine Transform (DCT-II) -> MFCC
        mfcc = scipy.fftpack.dct(log_mel, type=2, axis=0, norm='ortho')[:self.n_mfcc]
        
        # 5. Delta and Delta-Delta velocity/acceleration features
        delta1 = self._compute_delta(mfcc)
        delta2 = self._compute_delta(delta1)
        
        # 6. Standardize temporal length to target_frames (128)
        feats = np.stack([mfcc, delta1, delta2], axis=0)  # [3, 40, T]
        curr_frames = feats.shape[2]
        
        if curr_frames < self.target_frames:
            # Pad with zeros
            pad_width = self.target_frames - curr_frames
            feats = np.pad(feats, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        elif curr_frames > self.target_frames:
            # Center crop
            start = (curr_frames - self.target_frames) // 2
            feats = feats[:, :, start:start + self.target_frames]
            
        # 7. Normalize per channel
        for c in range(3):
            mean = np.mean(feats[c])
            std = np.std(feats[c]) + 1e-8
            feats[c] = (feats[c] - mean) / std
            
        return feats.astype(np.float32)


# =====================================================================
# 2. PyTorch Speech Emotion Recognition Dataset
# =====================================================================
class SpeechEmotionDataset(Dataset):
    """Dataset for speech emotion audio samples with on-the-fly MFCC caching."""

    def __init__(self, records: list, project_root: str, extractor: MFCCFeatureExtractor):
        self.records = records
        self.project_root = project_root
        self.extractor = extractor
        self.data = []
        self._prepare()

    def _prepare(self):
        for rec in self.records:
            full_path = os.path.join(self.project_root, rec["filepath"])
            audio, _ = sf.read(full_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            feats = self.extractor.extract(audio)
            label = EMOTION2IDX[rec["emotion"]]
            self.data.append((feats, label, rec))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feats, label, rec = self.data[idx]
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long), rec


# =====================================================================
# 3. Deep CRNN Model (2D-CNN + BiLSTM + Attention Temporal Pooling)
# =====================================================================
class AttentionPooling(nn.Module):
    """Self-Attention Temporal Pooling layer."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        # x: [batch, seq_len, hidden_dim]
        scores = self.attention(x)  # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch, seq_len, 1]
        context = torch.sum(x * weights, dim=1)  # [batch, hidden_dim]
        return context, weights


class SpeechEmotionCRNN(nn.Module):
    """
    CRNN architecture for Speech Emotion Recognition:
    - 2D-CNN: Extracts local spectro-temporal acoustic patterns
    - BiLSTM: Models long-range prosodic and emotional trajectories
    - Self-Attention: Focuses on emotionally salient speech frames
    """
    def __init__(self, num_classes: int = 5, in_channels: int = 3):
        super().__init__()
        
        # 1. 2D Convolutional Front-End
        self.conv_blocks = nn.Sequential(
            # Block 1: [3, 40, 128] -> [32, 20, 64]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Block 2: [32, 20, 64] -> [64, 10, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Block 3: [64, 10, 32] -> [128, 5, 32]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(0.2)
        )
        
        # 2. Temporal Sequence Flattening: 128 channels * 5 freq bins = 640 dim
        self.lstm_input_dim = 128 * 5
        
        # 3. Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 4. Self-Attention Temporal Pooling (BiLSTM output: 256 dim)
        self.attention = AttentionPooling(hidden_dim=256)
        
        # 5. Emotion Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: [batch, 3, 40, 128]
        conv_out = self.conv_blocks(x)  # [batch, 128, 5, 32]
        
        # Reshape for sequence model: [batch, time_steps=32, feature_dim=640]
        batch_size, channels, freq, time_steps = conv_out.shape
        seq_in = conv_out.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, channels * freq)
        
        lstm_out, _ = self.bilstm(seq_in)  # [batch, 32, 256]
        context, attn_weights = self.attention(lstm_out)  # [batch, 256]
        logits = self.classifier(context)  # [batch, num_classes]
        
        return logits, attn_weights


# =====================================================================
# 4. Training & Evaluation Pipeline
# =====================================================================
def train_and_evaluate(dataset_path: str, output_dir: str):
    """Trains the Speech Emotion CRNN and benchmarks on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(dataset_path)))
    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]
    
    print(f"Dataset: {len(records)} samples | Train: {len(train_records)} | Test: {len(test_records)}")
    
    extractor = MFCCFeatureExtractor()
    print("Extracting 40-band MFCCs + Delta + Delta-Delta acoustic features...")
    train_dataset = SpeechEmotionDataset(train_records, project_root, extractor)
    test_dataset = SpeechEmotionDataset(test_records, project_root, extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = SpeechEmotionCRNN(num_classes=len(EMOTIONS)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CRNN Model Initialized! Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    
    num_epochs = 25
    train_losses = []
    train_accs = []
    
    print("\nStarting CRNN Training (25 Epochs)...")
    print("-" * 65)
    start_train_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for feats, labels, _ in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:5.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
            
    train_duration = time.time() - start_train_time
    print(f"Training completed in {train_duration:.2f}s!")
    
    # -----------------------------------------------------------------
    # Test Evaluation
    # -----------------------------------------------------------------
    print("\nEvaluating on Held-Out Test Set (25 Samples)...")
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    latencies = []
    test_details = []
    
    with torch.no_grad():
        for feats, labels, recs in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            
            t0 = time.time()
            logits, _ = model(feats)
            lat = (time.time() - t0) * 1000.0 / len(labels)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            for i in range(len(labels)):
                true_lbl = labels[i].item()
                pred_lbl = preds[i].item()
                conf = probs[i, pred_lbl].item()
                y_true.append(true_lbl)
                y_pred.append(pred_lbl)
                y_probs.append(probs[i].cpu().numpy().tolist())
                latencies.append(lat)
                
                is_correct = (true_lbl == pred_lbl)
                rec_id = recs["id"][i] if isinstance(recs["id"], list) else recs[i]["id"]
                rec_fn = recs["filename"][i] if isinstance(recs["filename"], list) else recs[i]["filename"]
                test_details.append({
                    "id": rec_id,
                    "filename": rec_fn,
                    "ground_truth": IDX2EMOTION[true_lbl],
                    "predicted": IDX2EMOTION[pred_lbl],
                    "confidence": float(conf),
                    "is_correct": bool(is_correct),
                    "latency_ms": float(lat)
                })

    # Metrics calculation
    test_acc = (sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTIONS))))
    mean_latency = float(np.mean(latencies))
    
    print("=" * 75)
    print(f"Speech Emotion Recognition Test Accuracy : {test_acc:.2f}%")
    print(f"Macro Precision                         : {prec * 100:.2f}%")
    print(f"Macro Recall                            : {rec * 100:.2f}%")
    print(f"Macro F1-Score                          : {f1 * 100:.2f}%")
    print(f"Mean Inference Latency                  : {mean_latency:.2f} ms")
    print("=" * 75)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=3))

    # Save model weights
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "ser_crnn_model.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Trained model checkpoint saved to: {weights_path}")
    
    # Save benchmark results JSON
    results_json_path = os.path.join(output_dir, "emotion_results.json")
    results_payload = {
        "model_architecture": "CRNN (2D-CNN + BiLSTM + Attention)",
        "num_classes": len(EMOTIONS),
        "classes": EMOTIONS,
        "test_accuracy": test_acc,
        "macro_precision": float(prec * 100.0),
        "macro_recall": float(rec * 100.0),
        "macro_f1": float(f1 * 100.0),
        "mean_latency_ms": mean_latency,
        "confusion_matrix": conf_mat.tolist(),
        "test_samples": test_details
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Benchmark metrics saved to: {results_json_path}")
    
    # Generate Executive Diagnostic Visualizations
    generate_diagnostic_report(
        records=records,
        project_root=project_root,
        extractor=extractor,
        train_losses=train_losses,
        train_accs=train_accs,
        conf_mat=conf_mat,
        test_acc=test_acc,
        macro_f1=f1 * 100.0,
        output_dir=output_dir
    )


# =====================================================================
# 5. Diagnostic Visualization Generator
# =====================================================================
def generate_diagnostic_report(records: list, project_root: str, extractor: MFCCFeatureExtractor,
                               train_losses: list, train_accs: list, conf_mat: np.ndarray,
                               test_acc: float, macro_f1: float, output_dir: str):
    """Generates an executive 4-panel diagnostic chart for Speech Emotion Recognition."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor('#F8F9FA')

    # -------------------------------------------------------------
    # Panel 1: Multi-Emotion Waveform & Acoustic Envelope Comparison
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor('#FFFFFF')
    
    emotion_colors = {
        "neutral": "#64748B",
        "happy": "#F59E0B",
        "sad": "#3B82F6",
        "angry": "#EF4444",
        "fearful": "#8B5CF6"
    }
    
    # Grab one audio sample for each emotion from the same prompt
    offset = 0.0
    for emo in EMOTIONS:
        sample = next(r for r in records if r["emotion"] == emo)
        audio, _ = sf.read(os.path.join(project_root, sample["filepath"]))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # Display 1.5 seconds slice
        slice_len = min(len(audio), int(1.5 * 16000))
        t_axis = np.linspace(0, 1.5, slice_len)
        sig = audio[:slice_len] / (np.max(np.abs(audio[:slice_len])) + 1e-8) * 0.4
        
        ax1.plot(t_axis, sig + offset, color=emotion_colors[emo], linewidth=1.0, label=emo.title())
        offset += 0.95
        
    ax1.set_yticks(np.arange(len(EMOTIONS)) * 0.95)
    ax1.set_yticklabels([e.title() for e in EMOTIONS], fontsize=9, fontweight='bold')
    ax1.set_xlabel("Time (seconds)", fontsize=9)
    ax1.set_title("A) Comparative Speech Waveforms & Prosodic Envelopes", fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)

    # -------------------------------------------------------------
    # Panel 2: 40-Band MFCC Spectrograms Across Contrasting Emotions
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor('#FFFFFF')
    
    # Stack two contrasting MFCCs: Happy vs Sad
    sample_happy = next(r for r in records if r["emotion"] == "happy")
    sample_sad = next(r for r in records if r["emotion"] == "sad")
    
    audio_happy, _ = sf.read(os.path.join(project_root, sample_happy["filepath"]))
    audio_sad, _ = sf.read(os.path.join(project_root, sample_sad["filepath"]))
    
    feats_happy = extractor.extract(audio_happy)[0]  # MFCC channel
    feats_sad = extractor.extract(audio_sad)[0]
    
    # Concatenate vertically with a divider
    divider = np.zeros((3, feats_happy.shape[1]))
    stacked = np.vstack([feats_happy, divider, feats_sad])
    
    im = ax2.imshow(stacked, aspect='auto', cmap='magma', origin='lower')
    cbar = plt.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label("Standardized MFCC Value", fontsize=8)
    
    ax2.axhline(41.5, color='#00FFCC', linestyle='--', linewidth=1.5)
    ax2.text(5, 75, "Happy (High Pitch & Energy)", color='#FFFFFF', fontsize=9, fontweight='bold')
    ax2.text(5, 5, "Sad (Depressed Pitch & Formants)", color='#FFFFFF', fontsize=9, fontweight='bold')
    ax2.set_title("B) 40-Channel MFCC Spectrogram Comparison (Happy vs Sad)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("Time Frame Index (hop: 10 ms)", fontsize=9)
    ax2.set_ylabel("MFCC Coefficient Index", fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Confusion Matrix Heatmap
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor('#FFFFFF')
    
    # Normalize confusion matrix
    cm_norm = conf_mat.astype('float') / (conf_mat.sum(axis=1)[:, np.newaxis] + 1e-8)
    im3 = ax3.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1.0)
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.85)
    cbar3.set_label("Normalized Recognition Accuracy", fontsize=8)
    
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            val = conf_mat[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax3.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center", 
                     color=text_color, fontsize=8, fontweight='bold')
                     
    ax3.set_xticks(range(len(EMOTIONS)))
    ax3.set_yticks(range(len(EMOTIONS)))
    ax3.set_xticklabels([e.title() for e in EMOTIONS], fontsize=8)
    ax3.set_yticklabels([e.title() for e in EMOTIONS], fontsize=8)
    ax3.set_xlabel("Predicted Emotion", fontsize=9)
    ax3.set_ylabel("Ground Truth Emotion", fontsize=9)
    ax3.set_title(f"C) Emotion Confusion Matrix (Test Accuracy: {test_acc:.1f}%)", fontsize=11, fontweight='bold', pad=10)

    # -------------------------------------------------------------
    # Panel 4: CRNN Training Convergence Curves (Loss & Accuracy)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor('#FFFFFF')
    epochs = range(1, len(train_losses) + 1)
    
    line1 = ax4.plot(epochs, train_losses, color='#EF4444', linewidth=2.0, label='Training Loss')
    ax4.set_xlabel("Training Epoch", fontsize=9)
    ax4.set_ylabel("Cross-Entropy Loss", color='#EF4444', fontsize=9)
    ax4.tick_params(axis='y', labelcolor='#EF4444')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    ax4_twin = ax4.twinx()
    line2 = ax4_twin.plot(epochs, train_accs, color='#10B981', linewidth=2.0, linestyle='--', label='Training Accuracy (%)')
    ax4_twin.set_ylabel("Accuracy (%)", color='#10B981', fontsize=9)
    ax4_twin.tick_params(axis='y', labelcolor='#10B981')
    ax4_twin.set_ylim(0, 105)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right', fontsize=8)
    ax4.set_title(f"D) CRNN Optimization Dynamics (Final Macro F1: {macro_f1:.1f}%)", fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "emotion_recognition_report.png")
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic report successfully saved to: {chart_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "data", "dataset.json")
    output_dir = os.path.join(base_dir, "outputs")
    train_and_evaluate(dataset_path, output_dir)


if __name__ == "__main__":
    main()
