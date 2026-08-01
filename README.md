# 🎙️ Advanced Voice Studio

An AI-powered voice cloning and speech generation application built using **Gradio**, **F5-TTS**, **RVC (Retrieval-Based Voice Conversion)**, **Edge-TTS**, and **Whisper**. The application supports voice cloning, multilingual speech generation, podcast creation, dramatic storytelling, and audio preprocessing.

---

# Features

- Voice Cloning using F5-TTS
- Voice Conversion using RVC
- Multi-Speaker Podcast Generator
- Dramatic Story Narration
- Hindi / Urdu Speech Generation
- Audio Enhancement
- Background Noise Reduction
- Silence Removal
- Audio Normalization
- Automatic Audio Chunking
- Voice Training Dataset Preprocessing
- Docker Support

---

# Internship Tasks

This repository was developed as part of the **ZenvyroLabs AI Internship**.

## Task 1 — Dockerization ✅

Containerized the complete application using Docker.

Implemented:

- Dockerfile
- docker-compose.yml
- Persistent Docker Volumes
- Automatic dependency installation
- GPU-ready configuration (optional)

Persistent volumes include:

```
hf_cache
saved_voices
training_data
rvc_models
```

---

## Task 2 — Podcast Improvements

Improved podcast generation by fixing:

- pronunciation issues
- text processing
- speaker handling
- generation workflow

---

## Task 3 — Audio Training Pipeline

Enhanced preprocessing before training.

Pipeline includes:

- Silence Removal
- Noise Reduction
- Volume Normalization
- Audio Chunking
- Metadata Generation
- Automatic Sample Rate Conversion
- Mono Audio Conversion

Output structure:

```
training_data/
    session_x/
        chunk_001.wav
        chunk_002.wav
        ...
        metadata.csv
```

---

# Technologies

- Python 3.11
- Gradio
- PyTorch
- F5-TTS
- RVC
- Edge-TTS
- Whisper
- FFmpeg
- Pydub
- Noisereduce
- HuggingFace Transformers
- Docker
- Docker Compose

---

# Project Structure

```
.
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── hf_cache/
├── saved_voices/
├── training_data/
├── rvc_models/
├── outputs/
├── temp/
├── assets/
└── README.md
```

---

# Installation

Clone the repository.

```bash
git clone <repository-url>

cd AdvancedVoiceStudio
```

Create virtual environment.

```bash
python -m venv venv
```

Activate environment.

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Running without Docker

Launch the application.

```bash
python app.py
```

Open

```
http://localhost:7860
```

---

# Docker Setup

Build

```bash
docker compose build
```

Run

```bash
docker compose up
```

Run in background

```bash
docker compose up -d
```

Stop

```bash
docker compose down
```

---

# Required Models

Place RVC models inside

```
rvc_models/
```

Voice datasets should be stored inside

```
training_data/
```

Generated voices are saved in

```
saved_voices/
```

HuggingFace models are cached inside

```
hf_cache/
```

---

# Audio Preprocessing Pipeline

Input Audio

↓

Noise Reduction

↓

Silence Removal

↓

Normalization

↓

Mono Conversion

↓

16kHz Resampling

↓

Chunk Generation

↓

Metadata Creation

↓

Training Ready Dataset

---

# Supported Features

- English Voice Cloning
- Hindi Voice Cloning
- Urdu Voice Cloning
- Multi-Speaker Podcasts
- Story Narration
- Audio Cleaning
- Voice Training
- Voice Conversion

---

# Troubleshooting

## FFmpeg not found

Install FFmpeg and add it to your system PATH.

---

## CUDA not detected

Install a compatible version of PyTorch with CUDA support.

---

## RVC model missing

Download an RVC model and place it inside

```
rvc_models/
```

---

## Docker out of disk space

Check disk usage

```bash
docker system df
```

Remove unused volumes

```bash
docker volume prune
```

Remove unused images

```bash
docker image prune -a
```

---

## No space left on device

Clean temporary files

```bash
docker system prune -a
```

or free storage on the host machine before rebuilding.

---

# Future Improvements

- Real-time voice cloning
- Speaker diarization
- Better multilingual pronunciation
- Faster inference
- GPU optimization
- Batch voice training
- Cloud deployment

---

# Acknowledgements

- F5-TTS
- Retrieval-Based Voice Conversion (RVC)
- HuggingFace
- Gradio
- PyTorch
- Edge-TTS
- Whisper

---

# License

This project was developed for educational and internship purposes.
