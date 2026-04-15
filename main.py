from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from tqdm import tqdm
import torchaudio
import torch
import os
import sys
import subprocess
import json
import tempfile
import logging
from datetime import datetime
import numpy as np
import soundfile as sf

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

NUM_SPEAKERS = int(os.getenv("NUM_SPEAKERS", "2"))

INTERVIEWS_DIR = "interviews"
os.makedirs(INTERVIEWS_DIR, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_audio_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_audio_to_wav(input_path: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_file.name
    temp_file.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        temp_wav_path
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav_path


def load_audio_for_pyannote(path: str):

    waveform, sample_rate = sf.read(path)

    if len(waveform.shape) == 1:
        waveform = waveform[np.newaxis, :]
    else:
        waveform = waveform.T # (channels, time)

    return {
        "waveform": torch.tensor(waveform, dtype=torch.float32),
        "sample_rate": sample_rate
    }


def get_best_speaker(segment_start: float, segment_end: float, diarization) -> str:
    best_speaker = "Unknown"
    best_overlap = 0.0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(segment_start, turn.start)
        overlap_end = min(segment_end, turn.end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker


def main():
    filename = input("Enter the file name (inside interviews/): ").strip()
    audio_path = os.path.join(INTERVIEWS_DIR, filename)

    if not os.path.exists(audio_path):
        logger.error("File not found: %s", audio_path)
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_file = os.path.join(INTERVIEWS_DIR, f"{base_name}_{timestamp}.txt")
    cache_file = os.path.join(INTERVIEWS_DIR, f"{base_name}_whisper_cache.json")

    temp_wav_path = None

    try:
        logger.info("Analyzing file...")
        duration = get_audio_duration(audio_path)

        if os.path.exists(cache_file):
            logger.info("Loading transcription from cache...")
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            whisper_segments = cache_data["segments"]
            info_data = cache_data["info"]

        else:
            logger.info("Loading Whisper model...")
            model = WhisperModel("large-v3", device="cpu", compute_type="int8")

            logger.info("Starting transcription...")
            segments, info = model.transcribe(audio_path, language="fr")

            whisper_segments = []
            current_time = 0.0

            with tqdm(total=duration, unit="s", desc="Transcription") as pbar:
                for segment in segments:
                    whisper_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })

                    if segment.end > current_time:
                        pbar.update(segment.end - current_time)
                        current_time = segment.end

            info_data = {
                "language": info.language,
                "language_probability": info.language_probability
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "segments": whisper_segments,
                    "info": info_data
                }, f, ensure_ascii=False, indent=2)

            logger.info("Transcription saved to cache: %s", cache_file)

        logger.info("Loading diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )

        logger.info("Extracting temporary audio for diarization...")
        temp_wav_path = extract_audio_to_wav(audio_path)

        logger.info("Preloading audio into memory...")
        audio_input = load_audio_for_pyannote(temp_wav_path)

        logger.info("Running diarization...")
        diarization = pipeline(audio_input, num_speakers=NUM_SPEAKERS)

        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization
        elif isinstance(diarization, tuple):
            diarization = diarization[0]

        if not hasattr(diarization, "itertracks"):
            raise TypeError(f"Unexpected diarization type: {type(diarization)}")

        logger.info("Aligning transcription with speaker segments...")
        final_lines = []

        for segment in whisper_segments:
            seg_start = segment["start"]
            seg_end = segment["end"]
            text = segment["text"].strip()

            if not text:
                continue

            speaker_label = get_best_speaker(seg_start, seg_end, diarization)

            final_lines.append(
                f"[{format_time(seg_start)} - {format_time(seg_end)}] {speaker_label}: {text}"
            )

        final_text = "\n".join(final_lines)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)

        logger.info("Transcription completed successfully.")
        logger.info("Output file: %s", output_file)
        logger.info(
            "Detected language: %s (%.2f)",
            info_data["language"],
            info_data["language_probability"]
        )

    except subprocess.CalledProcessError:
        logger.error("FFmpeg/FFprobe error. Make sure ffmpeg and ffprobe are installed.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            logger.info("Temporary file removed: %s", temp_wav_path)


if __name__ == "__main__":
    main()