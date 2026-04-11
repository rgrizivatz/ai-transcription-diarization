# AI Transcription and Speaker Diarization

A simple Python tool to transcribe audio/video files and identify
speakers.

This project combines: - Whisper (via faster-whisper) for
transcription - Pyannote for speaker diarization

## Features

-   Automatic transcription of audio and video files
-   Speaker segmentation (who speaks when)
-   Timestamped output
-   Caching of transcription results to avoid recomputation
-   Configurable number of speakers via environment variables

## Requirements

-   Python 3.9+
-   ffmpeg and ffprobe installed
-   Hugging Face account with access token

## Installation

``` bash
git clone https://github.com/<your-username>/ai-transcription-diarization.git
cd ai-transcription-diarization

pip install -r requirements.txt
```

## Configuration

Set the required environment variables:

``` bash
export HF_TOKEN="your_huggingface_token"
export NUM_SPEAKERS=2
```

## Usage

``` bash
python main.py
```

Then provide the path to your audio or video file when prompted.

## Output

The script generates: - A `.txt` file with timestamps, speakers, and
transcription - A cache file (`*_whisper_cache.json`) for faster re-runs

## Notes

-   Speaker diarization quality depends on audio clarity
-   You can adjust the number of speakers using `NUM_SPEAKERS`
-   GPU support can be enabled in the code for faster processing

## Author

Copyright (c) Richard Grizivatz
