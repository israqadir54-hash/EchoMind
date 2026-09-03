# EchoMind
## A Vision-Language Based Approach for Memory Assistance in Alzheimer's Patients via Egocentric Image Streams

EchoMind is a Flask-based vision-language memory-assistance prototype. The supplied submission build keeps the original application source in `app.py` and uses:

- YOLO11x for object detection
- Qwen2.5-VL for detailed visual descriptions, with BLIP as a fallback
- CLIP for personal-object and personal-place visual matching
- Sentence-BERT for semantic retrieval
- SQLite as the persistent metadata database
- Optional ChromaDB migration support
- Browser upload, camera capture, GPS, and EXIF date/time handling

## Repository safety

This repository is prepared for a **public GitHub repository**. Real environment files, generated uploads, reference images, local database files, model weights, caches, and editor/runtime artifacts are excluded by `.gitignore`.

**Never commit API keys, passwords, tokens, service-account JSON, or other credentials.**

Use `.env.example` as the template for local configuration and create your own untracked `.env` when needed.

## Project structure

```text
EchoMind_GitHub_Submission/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── uploads/
│   └── .gitkeep
├── reference_images/
│   └── .gitkeep
└── echomind_database/
    └── .gitkeep
```

## Local setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and update values for your local environment.
4. Place the required YOLO11x model file at the project root as:

```text
yolo11x.pt
```

The application can still start without YOLO11x, but object detection will be unavailable until the model is present.

5. Run:

```bash
python app.py
```

6. Open:

```text
http://127.0.0.1:5000
```

The Hugging Face vision models used by the application may download automatically on first use, so the first startup can require internet access and substantial disk space.

## Important submission note

Do not commit generated user data or local model files to the public repository. The application creates its SQLite database and runtime image directories locally.
