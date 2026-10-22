"""
Music Genre Dataset Crawler & Audio Normalizer
Collects real music tracks from open-access repositories (Internet Archive / Creative Commons / Netlabels)
Standardizes them to 30.0s, 16kHz mono audio, and generates Kaggle-ready metadata and folder structure.
"""

import os
import sys
import io
import json
import time

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import soundfile as sf
import librosa


# 10 chuẩn thể loại GTZAN / Music Information Retrieval
GENRE_QUERIES = {
    "blues": '(genre:Blues OR subject:blues) AND mediatype:audio AND NOT title:podcast AND NOT subject:"Radio Program"',
    "classical": '(genre:Classical OR subject:classical OR composer:beethoven OR composer:bach OR composer:mozart) AND mediatype:audio AND NOT title:podcast',
    "country": '(genre:Country OR subject:country) AND mediatype:audio AND NOT title:podcast AND NOT subject:"Radio Program"',
    "disco": '(subject:disco OR genre:disco OR subject:"dance pop") AND mediatype:audio AND NOT title:podcast AND NOT subject:"Radio Program"',
    "hiphop": '(subject:"hip-hop" OR subject:"hip hop" OR genre:"Hip-Hop") AND mediatype:audio AND NOT title:podcast',
    "jazz": '(genre:Jazz OR subject:jazz) AND mediatype:audio AND NOT title:podcast AND NOT subject:"Radio Program"',
    "metal": '(subject:metal OR genre:Metal OR subject:"heavy metal") AND mediatype:audio AND NOT title:podcast',
    "pop": '(subject:pop OR genre:Pop OR subject:"indie pop") AND mediatype:audio AND NOT title:podcast AND NOT subject:"Radio Program"',
    "reggae": '(subject:reggae OR genre:Reggae OR subject:dub) AND mediatype:audio AND NOT title:podcast',
    "rock": '(genre:Rock OR subject:rock OR subject:"hard rock") AND mediatype:audio AND NOT title:podcast AND NOT subject:"Radio Program"'
}


class MusicGenreCrawler:
    def __init__(self, output_dir="kaggle_dataset", sample_rate=16000, duration=30.0, offset=10.0):
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio")
        self.sample_rate = sample_rate
        self.duration = duration
        self.offset = offset
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        os.makedirs(self.audio_dir, exist_ok=True)

    def search_archive_items(self, genre, max_items=20):
        """Tìm kiếm các item âm nhạc trên Internet Archive theo genre."""
        query = GENRE_QUERIES.get(genre, f"genre:{genre} AND mediatype:audio")
        encoded_query = urllib.parse.quote(query)
        url = (
            f"https://archive.org/advancedsearch.php?q={encoded_query}"
            f"&fl[]=identifier,title,creator,year,licenseurl"
            f"&sort[]=downloads+desc"
            f"&rows={max_items * 3}&output=json"
        )
        
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=12) as response:
                data = json.loads(response.read().decode("utf-8"))
                docs = data.get("response", {}).get("docs", [])
                return docs
        except Exception as e:
            print(f"[{genre.upper()}] Error searching Archive.org: {e}")
            return []

    def get_track_audio_url(self, identifier):
        """Lấy URL file MP3 phù hợp từ metadata của item."""
        meta_url = f"https://archive.org/metadata/{identifier}/files"
        try:
            req = urllib.request.Request(meta_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                files_data = json.loads(response.read().decode("utf-8"))
                files = files_data.get("result", [])
                
                # Ưu tiên các file MP3 có độ dài hợp lý (từ 25s đến 600s để tránh radio show hàng giờ)
                candidates = []
                for f in files:
                    name = f.get("name", "")
                    if name.lower().endswith(".mp3"):
                        # Tránh các file preview ngắn ngủi hoặc file 64kbps chất lượng kém nếu có file tốt hơn
                        length_sec = float(f.get("length", 0) or 0)
                        if length_sec > 25.0 and length_sec < 600.0:
                            candidates.append((f, 2))  # độ dài chuẩn
                        elif length_sec == 0:
                            candidates.append((f, 1))  # không rõ độ dài nhưng là mp3

                candidates.sort(key=lambda x: x[1], reverse=True)
                if candidates:
                    chosen_file = candidates[0][0]
                    filename = chosen_file["name"]
                    dl_url = f"https://archive.org/download/{identifier}/{urllib.parse.quote(filename)}"
                    return dl_url, filename
        except Exception as e:
            pass
        return None, None

    def download_and_process_track(self, item, genre, track_idx):
        """Tải, cắt chuẩn 30s, resample về 16kHz mono, lưu WAV và trích xuất metadata."""
        ident = item.get("identifier")
        title = item.get("title", f"Track {track_idx}")
        creator = item.get("creator", "Unknown Artist")
        if isinstance(creator, list):
            creator = ", ".join(str(c) for c in creator[:3])
            
        genre_dir = os.path.join(self.audio_dir, genre)
        os.makedirs(genre_dir, exist_ok=True)
        
        target_filename = f"{genre}_{track_idx:04d}.wav"
        target_filepath = os.path.join(genre_dir, target_filename)
        rel_filepath = os.path.join("audio", genre, target_filename).replace("\\", "/")

        if os.path.exists(target_filepath):
            return {
                "track_id": f"{genre}_{track_idx:04d}",
                "filename": target_filename,
                "file_path": rel_filepath,
                "genre": genre,
                "title": title,
                "artist": creator,
                "duration_seconds": self.duration,
                "sample_rate": self.sample_rate,
                "license": item.get("licenseurl", "Creative Commons / Public Domain"),
                "source_url": f"https://archive.org/details/{ident}"
            }

        audio_url, orig_name = self.get_track_audio_url(ident)
        if not audio_url:
            return None

        try:
            req = urllib.request.Request(audio_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                audio_bytes = resp.read()

            # Đọc và cắt chuẩn bằng librosa
            y, sr = librosa.load(
                io.BytesIO(audio_bytes),
                sr=self.sample_rate,
                mono=True,
                offset=self.offset,
                duration=self.duration
            )

            # Đảm bảo đủ độ dài 30 giây (nếu thiếu thì pad thêm số 0)
            target_samples = int(self.sample_rate * self.duration)
            if len(y) < target_samples:
                if len(y) < int(self.sample_rate * 5.0):
                    return None  # Quá ngắn
                y = np.pad(y, (0, target_samples - len(y)), mode="constant")
            else:
                y = y[:target_samples]

            # Peak normalization
            max_val = np.max(np.abs(y))
            if max_val > 1e-4:
                y = y / max_val * 0.95

            # Lưu ra file WAV 16-bit PCM
            sf.write(target_filepath, y, self.sample_rate, subtype="PCM_16")

            return {
                "track_id": f"{genre}_{track_idx:04d}",
                "filename": target_filename,
                "file_path": rel_filepath,
                "genre": genre,
                "title": title,
                "artist": creator,
                "duration_seconds": self.duration,
                "sample_rate": self.sample_rate,
                "license": item.get("licenseurl", "Creative Commons / Public Domain"),
                "source_url": f"https://archive.org/details/{ident}"
            }
        except Exception as e:
            return None

    def crawl_dataset(self, tracks_per_genre=5, max_workers=4):
        """Thu thập dataset cho toàn bộ 10 thể loại."""
        print("=" * 70)
        print("          BẮT ĐẦU CRAWL DATASET ÂM NHẠC CHO KAGGLE")
        print("=" * 70)
        print(f"Số thể loại       : {len(GENRE_QUERIES)}")
        print(f"Số bài mỗi thể loại: {tracks_per_genre}")
        print(f"Chuẩn hóa         : {self.duration}s, {self.sample_rate} Hz, Mono PCM WAV")
        print(f"Thư mục đích      : {self.output_dir}")
        print("=" * 70)

        all_metadata = []

        for genre in GENRE_QUERIES.keys():
            print(f"\n[Crawling Genre: {genre.upper()}] Tìm kiếm ứng viên...")
            docs = self.search_archive_items(genre, max_items=tracks_per_genre * 3)
            print(f"-> Tìm thấy {len(docs)} ứng viên khả dụng. Đang tải và xử lý...")

            downloaded_count = 0
            for doc in docs:
                if downloaded_count >= tracks_per_genre:
                    break
                track_idx = downloaded_count + 1
                meta = self.download_and_process_track(doc, genre, track_idx)
                if meta:
                    all_metadata.append(meta)
                    downloaded_count += 1
                    print(f"   ✓ [{genre}_{track_idx:04d}] {meta['title'][:40]} | {meta['artist'][:25]}")
                time.sleep(0.3)

            print(f"-> Hoàn thành thể loại {genre}: {downloaded_count}/{tracks_per_genre} bài.")

        # Xuất metadata.csv
        meta_df = pd.DataFrame(all_metadata)
        meta_csv_path = os.path.join(self.output_dir, "metadata.csv")
        meta_df.to_csv(meta_csv_path, index=False, encoding="utf-8")

        print("\n" + "=" * 70)
        print(f" CRAWL HOÀN TẤT: Tổng cộng {len(all_metadata)} bài hát đã được chuẩn hóa.")
        print(f" File metadata: {meta_csv_path}")
        print("=" * 70)
        return meta_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crawl Music Genre Dataset for Kaggle")
    parser.add_argument("--tracks_per_genre", type=int, default=5, help="Số lượng bài mỗi thể loại")
    parser.add_argument("--output_dir", type=str, default="kaggle_dataset", help="Thư mục xuất dataset")
    args = parser.parse_args()

    crawler = MusicGenreCrawler(output_dir=args.output_dir)
    crawler.crawl_dataset(tracks_per_genre=args.tracks_per_genre)
