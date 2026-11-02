"""
Dataset Generator for Speech Emotion Recognition (SER).
Synthesizes spoken audio across 5 emotional states:
['neutral', 'happy', 'sad', 'angry', 'fearful']
using acoustic prosody modulation (pitch shift, tempo stretching, spectral shaping,
harmonic overdrive, and tremor modulation) on 16kHz PCM audio.
"""

import os
import json
import random
import subprocess
import numpy as np
import soundfile as sf
import scipy.signal

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]

BASE_PROMPTS = [
    "The quarterly presentation is scheduled for next Monday morning.",
    "I cannot believe what just happened to our entire team today.",
    "Everything is completely out of control right now.",
    "We finally achieved our biggest breakthrough of the entire year.",
    "There is someone walking outside in the darkness near the building.",
    "I wonder if there is any reason to continue working on this project.",
    "Stop telling me what to do and listen to what I am saying.",
    "Look at this incredible surprise waiting for us right here.",
    "Please do not leave me alone in this frightening place.",
    "The weather forecast indicates standard seasonal conditions for tomorrow."
]


def modulate_emotion(audio: np.ndarray, sr: int, emotion: str) -> np.ndarray:
    """Applies acoustic and prosodic modulation corresponding to target emotion."""
    y = audio.copy()
    
    if emotion == "neutral":
        # Natural baseline speech, minimal acoustic coloring
        pass
        
    elif emotion == "happy":
        # High pitch (+4 semitones), fast tempo (1.20x), bright treble boost
        # Resample for pitch shift up
        factor_pitch = 2 ** (3.5 / 12.0)
        len_orig = len(y)
        # Shift pitch up by resampling
        y_shifted = scipy.signal.resample(y, int(len_orig / factor_pitch))
        # Accelerate tempo
        target_len = int(len_orig / 1.18)
        y = scipy.signal.resample(y_shifted, target_len)
        # Treble brightness boost (peaking EQ)
        b, a = scipy.signal.butter(2, 2500 / (sr / 2), btype='high')
        treble = scipy.signal.lfilter(b, a, y)
        y = y + 0.4 * treble
        
    elif emotion == "sad":
        # Low pitch (-4 semitones), slow tempo (0.80x), warm low-pass filter, energy decay
        factor_pitch = 2 ** (-3.5 / 12.0)
        len_orig = len(y)
        y_shifted = scipy.signal.resample(y, int(len_orig / factor_pitch))
        target_len = int(len_orig / 0.82)
        y = scipy.signal.resample(y_shifted, target_len)
        # Low-pass filter (muffle higher harmonics)
        b, a = scipy.signal.butter(2, 1200 / (sr / 2), btype='low')
        y = scipy.signal.lfilter(b, a, y) * 0.70
        # Gentle decaying envelope
        t_env = np.linspace(1.0, 0.65, len(y))
        y = y * t_env
        
    elif emotion == "angry":
        # Elevated pitch (+2.5 semitones), sharp attack, harmonic saturation, high RMS
        factor_pitch = 2 ** (2.5 / 12.0)
        len_orig = len(y)
        y_shifted = scipy.signal.resample(y, int(len_orig / factor_pitch))
        target_len = int(len_orig / 1.10)
        y = scipy.signal.resample(y_shifted, target_len)
        # Harmonic saturation / overdrive via hyperbolic tangent
        b_treble, a_treble = scipy.signal.butter(2, 1800 / (sr / 2), btype='high')
        treble = scipy.signal.lfilter(b_treble, a_treble, y)
        y = np.tanh(2.4 * (y + 0.5 * treble))
        
    elif emotion == "fearful":
        # Modulated jitter/tremor (7.5 Hz vibrato), elevated pitch (+2.5 semitones), turbulent breathiness
        factor_pitch = 2 ** (2.5 / 12.0)
        len_orig = len(y)
        y_shifted = scipy.signal.resample(y, int(len_orig / factor_pitch))
        target_len = int(len_orig / 1.05)
        y = scipy.signal.resample(y_shifted, target_len)
        # Amplitude tremor modulation (shivering effect)
        t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
        tremor = 1.0 + 0.35 * np.sin(2 * np.pi * 7.5 * t)
        # Subtle turbulent noise
        noise = np.random.normal(0, 0.025, len(y)).astype(np.float32)
        y = (y * tremor) + noise

    # Standardize amplitude to [-0.95, 0.95]
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = (y / max_amp) * 0.95
    return y.astype(np.float32)


def main():
    random.seed(42)
    np.random.seed(42)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    print("Synthesizing base speech prompts via Windows SAPI...")
    base_audios = []
    sr = 16000
    
    for i, prompt in enumerate(BASE_PROMPTS):
        raw_path = os.path.join(audio_dir, f"temp_base_{i}.wav")
        ps_script = f"""
        $voice = New-Object -ComObject SAPI.SpVoice
        $stream = New-Object -ComObject SAPI.SpFileStream
        $stream.Open('{raw_path.replace(os.sep, "/")}', 3, $false)
        $voice.AudioOutputStream = $stream
        $voice.Speak('{prompt}')
        $stream.Close()
        """
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, capture_output=True)
        
        audio, orig_sr = sf.read(raw_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if orig_sr != sr:
            audio = scipy.signal.resample(audio, int(len(audio) * sr / orig_sr))
            
        base_audios.append(audio.astype(np.float32))
        if os.path.exists(raw_path):
            os.remove(raw_path)
            
    print(f"Generated {len(base_audios)} base speech stems.")
    
    # Generate 20 samples per emotion (100 total samples)
    dataset_records = []
    sample_counter = 1
    
    print("Applying acoustic prosody modulation across 5 emotions...")
    for emotion in EMOTIONS:
        # Generate 2 variations per base prompt (20 samples per emotion)
        for var_idx in range(2):
            for prompt_idx, base_audio in enumerate(base_audios):
                sample_id = f"ser_{sample_counter:03d}"
                filename = f"sample_{sample_id}_{emotion}.wav"
                final_path = os.path.join(audio_dir, filename)
                
                # Apply slight perturbation to base audio for variation
                audio_input = base_audio.copy()
                if var_idx > 0:
                    # Slight speed jitter (±3%)
                    speed_factor = random.uniform(0.97, 1.03)
                    audio_input = scipy.signal.resample(audio_input, int(len(audio_input) / speed_factor))
                
                # Apply emotional modulation
                emotion_audio = modulate_emotion(audio_input, sr, emotion)
                
                sf.write(final_path, emotion_audio, sr, subtype="PCM_16")
                duration = len(emotion_audio) / sr
                
                # 75% train / 25% test split
                split = "train" if (sample_counter % 4 != 0) else "test"
                
                record = {
                    "id": sample_id,
                    "filename": filename,
                    "filepath": os.path.relpath(final_path, os.path.dirname(base_dir)).replace(os.sep, "/"),
                    "emotion": emotion,
                    "duration_sec": round(duration, 2),
                    "split": split,
                    "sample_rate": sr
                }
                dataset_records.append(record)
                sample_counter += 1
                
    dataset_json_path = os.path.join(base_dir, "dataset.json")
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, indent=2)
        
    num_train = sum(1 for r in dataset_records if r["split"] == "train")
    num_test = sum(1 for r in dataset_records if r["split"] == "test")
    print(f"\nGenerated {len(dataset_records)} emotion-labeled audio samples!")
    print(f"Split distribution: {num_train} train ({num_train/len(dataset_records)*100:.1f}%), {num_test} test ({num_test/len(dataset_records)*100:.1f}%)")
    print(f"Dataset catalog saved to: {dataset_json_path}")


if __name__ == "__main__":
    main()
