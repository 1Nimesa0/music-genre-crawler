"""
Music Genre Dataset EDA & Visualization Tool
Generates publication-quality visualizations for Kaggle dataset:
1. Class Distribution
2. Audio Waveforms per Genre
3. Log-Mel Spectrograms per Genre
4. Acoustic Feature Analysis (Spectral Centroid, Rolloff, ZCR, RMS Energy, MFCC Heatmap)
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

GENRE_COLORS = {
    "blues": "#3498db",
    "classical": "#9b59b6",
    "country": "#f39c12",
    "disco": "#e91e63",
    "hiphop": "#e74c3c",
    "jazz": "#1abc9c",
    "metal": "#34495e",
    "pop": "#2ecc71",
    "reggae": "#16a085",
    "rock": "#d35400"
}


def plot_class_distribution(meta_df, output_path="figures/01_class_distribution.png"):
    """Vẽ biểu đồ phân bố số lượng và tổng thời lượng của từng thể loại."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    genre_counts = meta_df["genre"].value_counts().sort_index()
    colors = [GENRE_COLORS.get(g, "#555555") for g in genre_counts.index]
    
    bars = ax.bar(genre_counts.index, genre_counts.values, color=colors, edgecolor="black", alpha=0.85, width=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{int(yval)} tracks", ha="center", va="bottom", fontweight="bold", fontsize=9)
    
    ax.set_title("Distribution of Music Genres in Dataset", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Music Genre", fontweight="bold")
    ax.set_ylabel("Number of 30s Audio Tracks", fontweight="bold")
    ax.set_ylim(0, max(genre_counts.values) * 1.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_waveforms(meta_df, base_dir=".", output_path="figures/02_waveforms_comparison.png"):
    """Vẽ sóng âm (Waveforms) mẫu cho các thể loại."""
    genres = sorted(meta_df["genre"].unique())
    n_genres = len(genres)
    cols = 2
    rows = (n_genres + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 2.2), dpi=250, sharex=True)
    axes = axes.flatten()

    for idx, genre in enumerate(genres):
        sample = meta_df[meta_df["genre"] == genre].iloc[0]
        audio_file = os.path.join(base_dir, sample["file_path"])
        y, sr = librosa.load(audio_file, sr=16000, duration=30.0)
        time_axis = np.linspace(0, len(y) / sr, len(y))

        ax = axes[idx]
        color = GENRE_COLORS.get(genre, "#333333")
        ax.plot(time_axis, y, color=color, alpha=0.8, linewidth=0.6)
        ax.set_title(f"{genre.upper()}: {sample['title'][:30]} ({sample['artist'][:20]})", fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

    for i in range(n_genres, len(axes)):
        fig.delaxes(axes[i])

    axes[min(n_genres - 1, len(axes) - 1)].set_xlabel("Time (seconds)", fontweight="bold")
    if n_genres > 1:
        axes[min(n_genres - 2, len(axes) - 2)].set_xlabel("Time (seconds)", fontweight="bold")

    fig.suptitle("Time-Domain Waveforms Across Music Genres (16kHz, 30s normalized)", fontsize=14, fontweight="bold", y=1.002)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_spectrograms(meta_df, base_dir=".", output_path="figures/03_spectrograms_comparison.png"):
    """Vẽ Log-Mel Spectrograms so sánh trực quan giữa các thể loại."""
    genres = sorted(meta_df["genre"].unique())
    n_genres = len(genres)
    cols = 2
    rows = (n_genres + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 2.6), dpi=250, sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, genre in enumerate(genres):
        sample = meta_df[meta_df["genre"] == genre].iloc[0]
        audio_file = os.path.join(base_dir, sample["file_path"])
        y, sr = librosa.load(audio_file, sr=16000, duration=30.0)

        # 64-Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
        S_dB = librosa.power_to_db(S, ref=np.max)

        ax = axes[idx]
        img = librosa.display.specshow(S_dB, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(f"{genre.upper()}: {sample['title'][:30]}", fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("Mel Freq (Hz)", fontsize=8)

    for i in range(n_genres, len(axes)):
        fig.delaxes(axes[i])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
    cbar_ax.set_ylabel("Power (dB)", fontweight="bold")

    fig.suptitle("Log-Mel Spectrograms Across Music Genres (64 Mels, n_fft=1024, hop=256)", fontsize=14, fontweight="bold", y=0.995)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_acoustic_features(meta_df, base_dir=".", output_path="figures/04_acoustic_features_comparison.png"):
    """Trích xuất và so sánh các đặc trưng âm học: Spectral Centroid, Rolloff, ZCR, RMS Energy."""
    feature_records = []
    for _, row in meta_df.iterrows():
        audio_file = os.path.join(base_dir, row["file_path"])
        if not os.path.exists(audio_file):
            continue
        y, sr = librosa.load(audio_file, sr=16000, duration=30.0)
        
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        
        feature_records.append({
            "genre": row["genre"],
            "spectral_centroid": centroid,
            "spectral_rolloff": rolloff,
            "zero_crossing_rate": zcr,
            "rms_energy": rms
        })

    feat_df = pd.DataFrame(feature_records)
    if feat_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=250)
    
    # 1. Spectral Centroid
    sns.boxplot(data=feat_df, x="genre", y="spectral_centroid", palette=GENRE_COLORS, ax=axes[0, 0], hue="genre", legend=False)
    axes[0, 0].set_title("Spectral Centroid (Frequency Brightness)", fontweight="bold")
    axes[0, 0].set_ylabel("Hz")
    axes[0, 0].tick_params(axis="x", rotation=30)
    
    # 2. Spectral Rolloff
    sns.boxplot(data=feat_df, x="genre", y="spectral_rolloff", palette=GENRE_COLORS, ax=axes[0, 1], hue="genre", legend=False)
    axes[0, 1].set_title("Spectral Rolloff (85% Energy Frequency)", fontweight="bold")
    axes[0, 1].set_ylabel("Hz")
    axes[0, 1].tick_params(axis="x", rotation=30)
    
    # 3. Zero-Crossing Rate
    sns.boxplot(data=feat_df, x="genre", y="zero_crossing_rate", palette=GENRE_COLORS, ax=axes[1, 0], hue="genre", legend=False)
    axes[1, 0].set_title("Zero-Crossing Rate (Percussiveness / Noise Ratio)", fontweight="bold")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].tick_params(axis="x", rotation=30)
    
    # 4. RMS Energy
    sns.boxplot(data=feat_df, x="genre", y="rms_energy", palette=GENRE_COLORS, ax=axes[1, 1], hue="genre", legend=False)
    axes[1, 1].set_title("Root-Mean-Square (RMS) Energy", fontweight="bold")
    axes[1, 1].set_ylabel("Energy Amplitude")
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.suptitle("Acoustic Feature Comparison Across Music Genres", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def run_all_visualizations(dataset_dir="."):
    meta_path = os.path.join(dataset_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} does not exist!")
        return
    meta_df = pd.read_csv(meta_path)
    print(f"Loaded metadata with {len(meta_df)} tracks.")
    
    plot_class_distribution(meta_df)
    plot_waveforms(meta_df, base_dir=dataset_dir)
    plot_spectrograms(meta_df, base_dir=dataset_dir)
    plot_acoustic_features(meta_df, base_dir=dataset_dir)
    print("All visualizations successfully generated in 'kaggle_dataset/figures/'!")


if __name__ == "__main__":
    run_all_visualizations()
