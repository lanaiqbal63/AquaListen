
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
import json

SR = 16000
SAMPLES = 48000
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="AquaListen",
    page_icon="🌊",
    layout="wide"
)

# Model architecture
class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = CNNBranch()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class AquaListen(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mel_branch = CNNBranch()
        self.mfcc_branch = CNNBranch()

        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, mel, mfcc):
        mel_embedding = self.mel_branch(mel)
        mfcc_embedding = self.mfcc_branch(mfcc)

        combined = torch.cat(
            [mel_embedding, mfcc_embedding],
            dim=1
        )

        return self.fusion(combined)


# Load model files
@st.cache_resource
def load_models():

    with open("AquaListen_Models/class_names.json") as f:
        class_names = json.load(f)

    with open("AquaListen_Models/normalization.json") as f:
        norm = json.load(f)

    baseline = BaselineCNN(len(class_names))
    aqua = AquaListen(len(class_names))

    baseline.load_state_dict(
        torch.load(
            "AquaListen_Models/baseline_model.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    aqua.load_state_dict(
        torch.load(
            "AquaListen_Models/aqualisten_model.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    baseline.eval()
    aqua.eval()

    return baseline, aqua, class_names, norm


# Extract features
def extract_features(audio, norm):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=64,
        n_fft=512,
        hop_length=256
    )

    mel = librosa.power_to_db(
        mel + 1e-9,
        ref=np.max
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=20,
        n_fft=512,
        hop_length=256
    )

    mel = (mel - norm["mel_mean"]) / norm["mel_std"]
    mfcc = (mfcc - norm["mfcc_mean"]) / norm["mfcc_std"]

    mel_tensor = torch.tensor(
        mel, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    mfcc_tensor = torch.tensor(
        mfcc, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return mel_tensor, mfcc_tensor, mel, mfcc


# Application interface
st.title("🌊 AquaListen")
st.subheader(
    "Noise-Robust Underwater Acoustic Classification "
    "Using Dual-Feature Fusion"
)

st.write(
    "Upload an underwater WAV recording to classify "
    "it using the Baseline CNN and AquaListen models."
)

uploaded_file = st.file_uploader(
    "Upload underwater audio (WAV)",
    type=["wav"]
)

if uploaded_file is not None:

    audio, sr = librosa.load(
        uploaded_file,
        sr=SR,
        mono=True
    )

    if len(audio) == 0:
        st.error("The uploaded audio is empty.")
        st.stop()

    # Standardize input duration to 3 seconds
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    elif len(audio) < SAMPLES:
        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    st.audio(uploaded_file)

    baseline, aqua, class_names, norm = load_models()

    mel_tensor, mfcc_tensor, mel, mfcc = extract_features(
        audio, norm
    )

    # Predictions
    with torch.no_grad():

        baseline_probs = torch.softmax(
            baseline(mel_tensor), dim=1
        )[0].numpy()

        aqua_probs = torch.softmax(
            aqua(mel_tensor, mfcc_tensor), dim=1
        )[0].numpy()

    baseline_index = int(np.argmax(baseline_probs))
    aqua_index = int(np.argmax(aqua_probs))

    st.divider()
    st.header("Classification Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline CNN")
        st.success(class_names[baseline_index])
        st.metric(
            "Top prediction score",
            f"{baseline_probs[baseline_index] * 100:.2f}%"
        )

    with col2:
        st.subheader("AquaListen")
        st.success(class_names[aqua_index])
        st.metric(
            "Top prediction score",
            f"{aqua_probs[aqua_index] * 100:.2f}%"
        )

    # Top 3 predictions
    st.header("Top 3 AquaListen Predictions")

    top_indices = np.argsort(aqua_probs)[::-1][:3]

    for index in top_indices:
        st.write(
            f"**{class_names[index]}** — "
            f"{aqua_probs[index] * 100:.2f}%"
        )
        st.progress(float(aqua_probs[index]))

    # Audio visualizations
    st.divider()
    st.header("Audio Visualizations")

    fig, ax = plt.subplots(figsize=(10, 3))
    time_axis = np.arange(len(audio)) / SR
    ax.plot(time_axis, audio)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(
            mel,
            aspect="auto",
            origin="lower"
        )
        ax.set_title("Mel Spectrogram")
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bands")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(
            mfcc,
            aspect="auto",
            origin="lower"
        )
        ax.set_title("MFCC Features")
        ax.set_xlabel("Time frames")
        ax.set_ylabel("MFCC coefficients")
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "Research prototype: predictions are limited to the "
        "six trained species and should not be treated as "
        "confirmed identification. Prediction scores are "
        "not necessarily calibrated probabilities."
    )
