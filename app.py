# EchoMind FINAL SUBMISSION BUILD
# Vision-Language Based Approach for Memory Assistance in Alzheimer's Patients
# via Egocentric Image Streams
#
# Safe design goals:
# - Original images are saved BEFORE AI processing.
# - SQLite is the authoritative persistent database for metadata.
# - Existing uploads/reference_images are never deleted automatically.
# - Existing ChromaDB data is imported on first run when possible.
# - YOLO11x full-frame precision detection; registered personal profiles handle fine-grained items separately.
# - BLIP image descriptions + object-crop descriptions.
# - CLIP matching for registered personal objects and places.
# - Sentence-BERT semantic retrieval + keyword/recency ranking.
# - Browser upload, camera capture, GPS, EXIF date/time.
# - Database viewer, delete, personal objects, personal places, chatbot.

import os
import re
import json
import uuid
import math
import base64
import io
import sqlite3
import threading
from datetime import datetime
import traceback
from pathlib import Path

from flask import Flask, request, render_template_string, redirect, url_for, jsonify, send_from_directory, send_file, make_response
from PIL import Image, ExifTags, ImageEnhance, ImageFilter

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except Exception:
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None


# ============================================================
# APP / STORAGE
# ============================================================

APP_VERSION = "29.0 — Object-First Deterministic Memory Retrieval"
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REFERENCE_DIR = BASE_DIR / "reference_images"
DATA_DIR = BASE_DIR / "echomind_database"
SQLITE_PATH = DATA_DIR / "echomind.sqlite3"
MIGRATION_FLAG = DATA_DIR / ".chroma_migrated_v2"

for folder in (UPLOAD_DIR, REFERENCE_DIR, DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("ECHOMIND_SECRET", "echomind-local-demo-secret")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
app.jinja_env.globals["memory_time_label"] = lambda m: ""  # replaced below after helper definition

DB_LOCK = threading.RLock()


def db():
    conn = sqlite3.connect(str(SQLITE_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with DB_LOCK:
        conn = db()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            image TEXT NOT NULL,
            image_url TEXT NOT NULL,
            original_filename TEXT DEFAULT '',
            capture_timestamp TEXT NOT NULL,
            capture_time_source TEXT DEFAULT 'Unknown',
            date TEXT DEFAULT '',
            day TEXT DEFAULT '',
            time TEXT DEFAULT '',
            original_exif_time TEXT DEFAULT '',
            latitude REAL,
            longitude REAL,
            gps_accuracy REAL,
            location_source TEXT DEFAULT 'Unavailable',
            area TEXT DEFAULT '',
            semantic_place TEXT DEFAULT '',
            scene TEXT DEFAULT '',
            description TEXT DEFAULT '',
            user_description TEXT DEFAULT '',
            clip_regions_json TEXT DEFAULT '[]',
            objects_json TEXT DEFAULT '[]',
            personal_objects_json TEXT DEFAULT '[]',
            nearby_objects_json TEXT DEFAULT '[]',
            place_json TEXT DEFAULT '{}',
            relationships_json TEXT DEFAULT '[]',
            tags_json TEXT DEFAULT '[]',
            status TEXT DEFAULT 'saved',
            search_text TEXT DEFAULT '',
            embedding_json TEXT DEFAULT '[]',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS personal_objects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            owner TEXT DEFAULT '',
            location TEXT DEFAULT '',
            category TEXT DEFAULT '',
            description TEXT DEFAULT '',
            reference_image TEXT NOT NULL,
            clip_embedding_json TEXT DEFAULT '[]',
            reference_image_blob BLOB,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS personal_places (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            owner TEXT DEFAULT '',
            room TEXT DEFAULT '',
            category TEXT DEFAULT '',
            description TEXT DEFAULT '',
            reference_image TEXT NOT NULL,
            clip_embedding_json TEXT DEFAULT '[]',
            reference_image_blob BLOB,
            created_at TEXT NOT NULL
        );
        """)
        place_cols = {r[1] for r in conn.execute("PRAGMA table_info(personal_places)").fetchall()}
        object_cols = {r[1] for r in conn.execute("PRAGMA table_info(personal_objects)").fetchall()}
        memory_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "category" not in place_cols:
            conn.execute("ALTER TABLE personal_places ADD COLUMN category TEXT DEFAULT ''")
        if "location" not in object_cols:
            conn.execute("ALTER TABLE personal_objects ADD COLUMN location TEXT DEFAULT ''")
        if "reference_image_blob" not in object_cols:
            conn.execute("ALTER TABLE personal_objects ADD COLUMN reference_image_blob BLOB")
        if "reference_image_blob" not in place_cols:
            conn.execute("ALTER TABLE personal_places ADD COLUMN reference_image_blob BLOB")
        if "user_description" not in memory_cols:
            conn.execute("ALTER TABLE memories ADD COLUMN user_description TEXT DEFAULT ''")
        if "clip_regions_json" not in memory_cols:
            conn.execute("ALTER TABLE memories ADD COLUMN clip_regions_json TEXT DEFAULT '[]'")
        if "nearby_objects_json" not in memory_cols:
            conn.execute("ALTER TABLE memories ADD COLUMN nearby_objects_json TEXT DEFAULT '[]'")
        if "capture_time_source" not in memory_cols:
            conn.execute("ALTER TABLE memories ADD COLUMN capture_time_source TEXT DEFAULT 'Unknown'")
        conn.commit()
        conn.close()


init_db()


def verify_database_schema():
    """Fail early with a clear message if the persistent DB schema cannot support the build."""
    with DB_LOCK:
        conn=db()
        try:
            for table, required in {
                'memories': {'id','image','description','user_description','clip_regions_json','nearby_objects_json','capture_time_source','created_at'},
                'personal_objects': {'id','name','owner','location','category','description','reference_image','clip_embedding_json','reference_image_blob','created_at'},
                'personal_places': {'id','name','owner','room','category','description','reference_image','clip_embedding_json','reference_image_blob','created_at'},
            }.items():
                actual={r[1] for r in conn.execute(f'PRAGMA table_info({table})').fetchall()}
                missing=required-actual
                if missing:
                    raise RuntimeError(f'Database table {table} is missing required columns: {", ".join(sorted(missing))}')
        finally:
            conn.close()


verify_database_schema()


# ============================================================
# MODEL LOADING — each component can fail without destroying DB
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 78)
print("ECHOMIND", APP_VERSION)
print("Device:", DEVICE)
print("Database:", SQLITE_PATH)
print("=" * 78)


yolo_model = None
blip_processor = None
blip_model = None
clip_processor = None
clip_model = None
embedding_model = None
qwen_processor = None
qwen_model = None
QWEN_MODEL_NAME = os.environ.get("ECHOMIND_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")


def load_models():
    global yolo_model, blip_processor, blip_model, clip_processor, clip_model, embedding_model, qwen_processor, qwen_model

    try:
        print("\n[1/4] Loading YOLO11x...")
        yolo_path = BASE_DIR / "yolo11x.pt"
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO11x model not found at {yolo_path}. Put yolo11x.pt beside this Python file.")
        yolo_model = YOLO(str(yolo_path))
        print("YOLO11x ready.")
    except Exception as e:
        print("YOLO warning:", e)

    try:
        print("\n[2/4] Loading BLIP Large...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model.to(DEVICE)
        blip_model.eval()
        print("BLIP Large ready.")
    except Exception as e:
        print("BLIP warning:", e)

    try:
        print("\n[3/4] Loading CLIP...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.to(DEVICE)
        clip_model.eval()
        print("CLIP ready.")
    except Exception as e:
        print("CLIP warning:", e)

    try:
        print("\n[4/4] Loading Sentence-BERT...")
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
        print("Sentence-BERT ready.")
    except Exception as e:
        print("Sentence-BERT warning:", e)

    # Qwen2.5-VL is the primary image-description model. BLIP remains a fallback
    # so EchoMind can still save memories when the larger VLM is unavailable.
    if AutoProcessor is not None and Qwen2_5_VLForConditionalGeneration is not None:
        try:
            print("\n[5/5] Loading Qwen2.5-VL...", QWEN_MODEL_NAME)
            qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_NAME,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="sdpa"
            )
            qwen_model.eval()
            print("Qwen2.5-VL ready.")
        except Exception as e:
            qwen_processor = None
            qwen_model = None
            print("Qwen warning:", e)
    else:
        print("Qwen warning: Qwen2.5-VL classes are unavailable in this Transformers installation.")


if os.environ.get("ECHOMIND_SKIP_MODELS") != "1":
    load_models()


# ============================================================
# HELPERS
# ============================================================

def _file_sha1(path):
    """Return a stable SHA-1 for exact-image deduplication."""
    import hashlib
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def now_local():
    return datetime.now().astimezone()


def parse_client_datetime(value):
    """Parse EchoMind's browser/device LOCAL wall-clock time exactly.

    The UI deliberately sends local date/time components without a timezone.
    That value is the authoritative save/capture timestamp for display.
    We never convert it through the server timezone and never substitute EXIF.
    """
    raw=str(value or '').strip()
    if raw:
        # Preferred format: YYYY-MM-DDTHH:MM:SS (no timezone suffix).
        try:
            return datetime.fromisoformat(raw[:19])
        except Exception:
            pass
        # Backward compatibility for older records that contain a timezone.
        try:
            return datetime.fromisoformat(raw.replace('Z','+00:00')).replace(tzinfo=None)
        except Exception:
            pass
    return datetime.now().replace(microsecond=0)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def json_load(value, default):
    try:
        if value is None or value == "":
            return default
        return json.loads(value)
    except Exception:
        return default


def json_dump(value):
    return json.dumps(value, ensure_ascii=False)


def cosine(a, b):
    """Cosine similarity that safely accepts Python lists or torch tensors."""
    try:
        if a is None or b is None:
            return -1.0
        if hasattr(a, "detach"):
            a = a.detach().float().flatten().cpu()
        if hasattr(b, "detach"):
            b = b.detach().float().flatten().cpu()
        if len(a) == 0 or len(b) == 0 or len(a) != len(b):
            return -1.0
        if hasattr(a, "tolist"):
            a = a.tolist()
        if hasattr(b, "tolist"):
            b = b.tolist()
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) * float(x) for x in a))
        nb = math.sqrt(sum(float(y) * float(y) for y in b))
        if na == 0.0 or nb == 0.0:
            return -1.0
        return dot / (na * nb)
    except Exception:
        return -1.0


def load_image(path):
    return Image.open(path).convert("RGB")


def secure_image_extension(filename):
    ext = Path(filename or "").suffix.lower()
    return ext if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"} else ".jpg"


def save_uploaded_file(file):
    """Save original image immediately. This function runs before AI."""
    if not file or not file.filename:
        raise ValueError("No image was selected.")
    filename = uuid.uuid4().hex + secure_image_extension(file.filename)
    path = UPLOAD_DIR / filename
    file.save(str(path))
    if not path.exists() or path.stat().st_size == 0:
        raise IOError("The uploaded image was not saved correctly.")
    return filename, path


def save_base64_image(data_url):
    if not data_url or "," not in data_url:
        raise ValueError("Camera image data is missing.")
    _, encoded = data_url.split(",", 1)
    try:
        raw = base64.b64decode(encoded, validate=False)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError("Camera image could not be decoded.") from e
    filename = uuid.uuid4().hex + ".jpg"
    path = UPLOAD_DIR / filename
    image.save(str(path), "JPEG", quality=96)
    return filename, path


def dms_to_decimal(value, ref):
    try:
        nums = [float(x) for x in value]
        out = nums[0] + nums[1] / 60 + nums[2] / 3600
        if str(ref).upper() in {"S", "W"}:
            out *= -1
        return out
    except Exception:
        return None


def parse_exif_datetime(value):
    """Parse common EXIF date formats without inventing a date."""
    if not value:
        return None
    text=str(value).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z"):
        try:
            dt=datetime.strptime(text, fmt)
            return dt
        except Exception:
            pass
    return None


def get_exif_metadata(path):
    out = {"taken_at": "", "latitude": None, "longitude": None}
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return out
            date_value = exif.get(36867) or exif.get(306)
            if date_value:
                out["taken_at"] = str(date_value)
            gps = exif.get_ifd(34853) if hasattr(exif, "get_ifd") else None
            if gps:
                lat = gps.get(2)
                lat_ref = gps.get(1)
                lon = gps.get(4)
                lon_ref = gps.get(3)
                if lat and lon:
                    out["latitude"] = dms_to_decimal(lat, lat_ref)
                    out["longitude"] = dms_to_decimal(lon, lon_ref)
    except Exception as e:
        print("EXIF warning:", e)
    return out


# ============================================================
# LOCATION
# ============================================================

geocoder = None
if Nominatim:
    try:
        geocoder = Nominatim(user_agent="EchoMind-Submission-2.0")
    except Exception:
        geocoder = None


def get_area_name(latitude, longitude):
    if latitude is None or longitude is None:
        return "Location unavailable"
    if geocoder is None:
        return f"GPS {latitude:.6f}, {longitude:.6f}"
    try:
        loc = geocoder.reverse((latitude, longitude), exactly_one=True, language="en", timeout=5)
        if loc:
            return str(loc.address)
    except Exception as e:
        print("Geocoding warning:", e)
    return f"GPS {latitude:.6f}, {longitude:.6f}"


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def enhance_for_detection(path):
    """Create a separate working copy; never replace the original."""
    working = UPLOAD_DIR / (Path(path).stem + "_processed.jpg")
    try:
        image = load_image(path)
        image = ImageEnhance.Sharpness(image).enhance(1.35)
        image = ImageEnhance.Contrast(image).enhance(1.05)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=3))
        image.save(str(working), "JPEG", quality=95, optimize=True)
        return working
    except Exception as e:
        print("Enhancement warning:", e)
        return Path(path)


def make_tiles(image, rows=3, cols=3, overlap=0.18):
    w, h = image.size
    tiles = []
    tw = w / cols
    th = h / rows
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tw - tw * overlap))
            y1 = max(0, int(r * th - th * overlap))
            x2 = min(w, int((c + 1) * tw + tw * overlap))
            y2 = min(h, int((r + 1) * th + th * overlap))
            tiles.append((image.crop((x1, y1, x2, y2)), x1, y1))
    return tiles


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ab = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union else 0.0


def detect_objects_precisely(path):
    """High-precision object detection for judge-facing output.

    Only the original full image is used for generic object detection. Tiled
    inference is intentionally disabled here because overlapping tiles can
    produce false labels for unrelated regions. Fine-grained personal objects
    are handled separately by the personal-profile matcher.
    """
    if yolo_model is None:
        return []
    image = load_image(path)
    W, H = image.size
    raw = []
    try:
        results = yolo_model.predict(source=image, imgsz=1280, conf=0.55, iou=0.55, max_det=80, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().tolist()]
                x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
                y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                name = str(yolo_model.names.get(cls, cls)).strip()
                if not name:
                    continue
                raw.append({"name": name, "confidence": round(confidence, 4), "bbox": [x1, y1, x2, y2], "source": "full_image"})
    except Exception as e:
        print("YOLO inference warning:", e)
        return []

    raw.sort(key=lambda x: float(x.get("confidence", 0)), reverse=True)
    final = []
    for d in raw:
        duplicate = False
        for existing in final:
            if norm_text(d["name"]) == norm_text(existing["name"]) and iou(d["bbox"], existing["bbox"]) >= 0.50:
                duplicate = True
                break
        if not duplicate:
            final.append(d)
        if len(final) >= 50:
            break

    for d in final:
        x1, y1, x2, y2 = d["bbox"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        d["center"] = [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)]
        d["area"] = round(area, 2)
        d["relative_size"] = round(area / max(1, W * H), 6)
        d["position"] = relative_position(d["center"], W, H)
        d.setdefault("description", "")
    return final


def relative_position(center, W, H):
    x, y = center
    horiz = "left" if x < W / 3 else "right" if x > 2 * W / 3 else "center"
    vert = "top" if y < H / 3 else "bottom" if y > 2 * H / 3 else "middle"
    return f"{vert}-{horiz}"


def crop_object(image, bbox, padding=0.15):
    W, H = image.size
    x1, y1, x2, y2 = bbox
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    x1 = max(0, int(x1 - pw)); y1 = max(0, int(y1 - ph))
    x2 = min(W, int(x2 + pw)); y2 = min(H, int(y2 + ph))
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


# ============================================================
# VISION-LANGUAGE
# ============================================================

def blip_caption(image, prompt=None):
    if blip_model is None or blip_processor is None:
        return "Visual description unavailable because the BLIP model is not loaded."
    try:
        kwargs = {"images": image, "return_tensors": "pt"}
        if prompt:
            kwargs["text"] = prompt
        inputs = blip_processor(**kwargs)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_new_tokens=90, num_beams=5, repetition_penalty=1.12, no_repeat_ngram_size=3)
        return blip_processor.decode(output[0], skip_special_tokens=True).strip()
    except Exception as e:
        print("BLIP warning:", e)
        return ""



# ============================================================
# STRICT VISUAL DETECTION VERIFICATION
# ============================================================
# YOLO is allowed to propose candidates, but low-confidence / semantically
# inconsistent labels are NOT exposed to the user. BLIP checks the actual crop
# belonging to each YOLO box. The rest of EchoMind (nearby objects, spatial
# relations, personal matching and retrieval) works from this verified list.

DETECTION_ALIASES = {
    "person": {"person", "human"},
    "bicycle": {"bicycle", "bike"},
    "car": {"car", "automobile", "vehicle"},
    "motorcycle": {"motorcycle", "motorbike"},
    "airplane": {"airplane", "plane"},
    "bus": {"bus"},
    "train": {"train"},
    "truck": {"truck"},
    "boat": {"boat"},
    "traffic light": {"traffic light", "traffic signal"},
    "fire hydrant": {"fire hydrant"},
    "stop sign": {"stop sign"},
    "parking meter": {"parking meter"},
    "bench": {"bench"},
    "bird": {"bird"},
    "cat": {"cat"},
    "dog": {"dog"},
    "horse": {"horse"},
    "sheep": {"sheep"},
    "cow": {"cow"},
    "elephant": {"elephant"},
    "bear": {"bear"},
    "zebra": {"zebra"},
    "giraffe": {"giraffe"},
    "backpack": {"backpack", "school bag"},
    "umbrella": {"umbrella"},
    "handbag": {"handbag", "purse"},
    "tie": {"tie", "necktie"},
    "suitcase": {"suitcase", "luggage"},
    "frisbee": {"frisbee"},
    "skis": {"ski", "skis"},
    "snowboard": {"snowboard"},
    "sports ball": {"ball", "sports ball"},
    "kite": {"kite"},
    "baseball bat": {"baseball bat", "bat"},
    "baseball glove": {"baseball glove", "glove"},
    "skateboard": {"skateboard"},
    "surfboard": {"surfboard"},
    "tennis racket": {"tennis racket"},
    "bottle": {"bottle", "water bottle", "container"},
    "wine glass": {"wine glass", "glass"},
    "cup": {"cup", "mug"},
    "fork": {"fork"},
    "knife": {"knife"},
    "spoon": {"spoon"},
    "bowl": {"bowl"},
    "banana": {"banana"},
    "apple": {"apple"},
    "sandwich": {"sandwich"},
    "orange": {"orange"},
    "broccoli": {"broccoli"},
    "carrot": {"carrot"},
    "hot dog": {"hot dog"},
    "pizza": {"pizza"},
    "donut": {"donut", "doughnut"},
    "cake": {"cake"},
    "chair": {"chair", "office chair", "seat"},
    "couch": {"couch", "sofa"},
    "potted plant": {"plant", "potted plant"},
    "bed": {"bed"},
    "dining table": {"dining table", "table"},
    "toilet": {"toilet"},
    "tv": {"tv", "television", "monitor"},
    "laptop": {"laptop", "laptop computer"},
    "mouse": {"mouse", "computer mouse"},
    "remote": {"remote", "remote control"},
    "keyboard": {"keyboard", "computer keyboard"},
    "cell phone": {"phone", "cell phone", "smartphone", "mobile phone"},
    "microwave": {"microwave"},
    "oven": {"oven"},
    "toaster": {"toaster"},
    "sink": {"sink"},
    "refrigerator": {"refrigerator", "fridge"},
    "book": {"book", "notebook", "textbook", "manual"},
    "clock": {"clock", "wall clock"},
    "vase": {"vase"},
    "scissors": {"scissors", "scissor"},
    "teddy bear": {"teddy bear", "stuffed animal"},
    "hair drier": {"hair dryer", "hair drier"},
    "toothbrush": {"toothbrush"},
}

COCO_LABELS = list(DETECTION_ALIASES.keys())


def _label_supported_by_caption(label, caption):
    label_n = norm_text(label)
    caption_n = norm_text(caption)
    if not label_n or not caption_n:
        return False
    aliases = DETECTION_ALIASES.get(label_n, {label_n})
    return any(norm_text(alias) in caption_n for alias in aliases)


def _clip_text_features(texts):
    if clip_model is None or clip_processor is None or not texts:
        return None
    try:
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)
        outputs = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        features = clip_model.text_projection(outputs.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return features
    except Exception as e:
        print("CLIP text feature warning:", e)
        return None


def _clip_classify_crop(crop, label):
    """Conservative CLIP class check for one YOLO crop.
    The crop must rank the detector's class above a broad set of common alternatives.
    This prevents a detector hallucination such as toilet/knife/tie on a desk scene from
    reaching the user-facing evidence panel."""
    if clip_model is None or clip_processor is None:
        return None
    try:
        prompts = [f"a clear photo of a {name}" for name in COCO_LABELS]
        text_features = _clip_text_features(prompts)
        if text_features is None:
            return None
        with torch.no_grad():
            image_features = _clip_image_tensor([crop])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sims = (image_features @ text_features.T)[0]
        order = torch.argsort(sims, descending=True).detach().cpu().tolist()
        label_n = norm_text(label)
        label_aliases = {norm_text(x) for x in DETECTION_ALIASES.get(label_n, {label_n})}
        rank = None
        label_score = None
        top_items = []
        for idx in order:
            candidate = COCO_LABELS[idx]
            sc = float(sims[idx].detach().cpu())
            top_items.append((candidate, sc))
            if norm_text(candidate) in label_aliases or any(norm_text(a) == norm_text(candidate) for a in label_aliases):
                rank = len(top_items)
                label_score = sc
                break
        top1, top1_score = top_items[0] if top_items else (None, -1.0)
        second_score = top_items[1][1] if len(top_items) > 1 else -1.0
        # Accept top-1 or, at most, top-3 when the requested label is a known semantic synonym.
        accepted_rank = rank is not None and rank <= 3
        margin = label_score - second_score if label_score is not None else -1.0
        return {
            "rank": rank,
            "label_score": label_score,
            "top1": top1,
            "top1_score": top1_score,
            "margin": margin,
            "accepted_rank": accepted_rank,
        }
    except Exception as e:
        print("CLIP crop classification warning:", e)
        return None


def _dedupe_verified_detections(detections):
    detections = sorted(detections, key=lambda d: float(d.get('confidence', 0) or 0), reverse=True)
    final = []
    for d in detections:
        same = False
        for e in final:
            if norm_text(d.get('name','')) == norm_text(e.get('name','')) and iou(d.get('bbox', []), e.get('bbox', [])) >= 0.45:
                same = True
                break
        if not same:
            final.append(d)
    return final[:50]


def verify_yolo_detections(image, detections, max_candidates=40):
    """Strictly verify YOLO detections before they enter the database/UI.
    Missing an uncertain object is preferred over displaying a false object."""
    if not detections:
        return []
    verified=[]
    blip_ok=blip_model is not None and blip_processor is not None
    clip_ok=clip_model is not None and clip_processor is not None
    W,H=image.size
    image_area=float(max(1,W*H))
    for d in sorted(detections,key=lambda x:float(x.get('confidence',0) or 0),reverse=True)[:max_candidates]:
        conf=float(d.get('confidence',0) or 0)
        if d.get('source')!='full_image' or conf<0.60:
            continue
        bbox=d.get('bbox') or []
        if len(bbox)!=4:
            continue
        x1,y1,x2,y2=map(float,bbox)
        bw,bh=max(0.0,x2-x1),max(0.0,y2-y1)
        rel_area=(bw*bh)/image_area
        if rel_area<0.002 and conf<0.80:
            continue
        if rel_area>0.85 and conf<0.85:
            continue
        label=str(d.get('name','')).strip()
        if not label:
            continue
        crop=crop_object(image,bbox,0.03)
        clip_check=_clip_classify_crop(crop,label) if clip_ok else None
        if clip_ok and clip_check is not None:
            if not clip_check.get('accepted_rank'):
                continue
            rank=clip_check.get('rank')
            score=clip_check.get('label_score')
            if rank!=1 and (rank is None or rank>3 or score is None or score<0.31):
                continue
            obvious_false={'toilet','knife','tie','cat','dog','tv','remote','laptop','keyboard','chair','bottle','mouse','book','cell phone'}
            top1=norm_text(clip_check.get('top1') or '')
            if label.lower() not in obvious_false and top1 in obvious_false and rank!=1:
                continue
        caption=''
        if blip_ok:
            try:
                caption=blip_caption(crop).strip()
            except Exception:
                caption=''
            if caption and not _label_supported_by_caption(label,caption):
                if clip_check is None or clip_check.get('rank')!=1 or float(clip_check.get('label_score') or 0)<0.34:
                    continue
                caption=''
        item=dict(d)
        item['description']=caption
        item['verification']='verified'
        item['verification_reason']='full-frame YOLO + CLIP crop verification'
        item['clip_class_rank']=clip_check.get('rank') if clip_check else None
        item['clip_class_score']=round(float(clip_check.get('label_score')),4) if clip_check and clip_check.get('label_score') is not None else None
        item['clip_class_top1']=clip_check.get('top1') if clip_check else None
        verified.append(item)
    return _dedupe_verified_detections(verified)


def qwen_image_description(image):
    """Generate a detailed, evidence-grounded description with Qwen2.5-VL.

    Qwen is explicitly told to identify ordinary objects and technical/electronic
    components (for example LED, resistor, potentiometer, PCB, connector, switch,
    capacitor, sensor, display and measuring equipment) only when visually supported.
    User-entered text is NOT fed into this description, preventing manual notes from
    being copied into the AI-generated description.
    """
    if qwen_model is None or qwen_processor is None:
        return ""
    prompt = """
Analyze this image carefully as a visual-memory assistant.

Write ONE natural, logical description of what is actually visible in the image.
Use concrete visual evidence, not generic filler.

You must:
1. Identify the main scene/object and its context from the pixels only.
2. Identify clearly visible objects and describe their appearance, relative position,
   and relevant physical features.
3. Pay special attention to technical and electronic items. When visually supported,
   distinguish components such as LED, resistor, potentiometer, capacitor, diode,
   transistor, IC/chip, PCB/prototype board, jumper wire, connector, switch,
   push button, sensor, motor, display, multimeter, oscilloscope, power supply,
   breadboard, relay, potentiometer knobs, terminals and other laboratory/electronics hardware.
4. Use physically and spatially logical lab context: electronics/lab components are
   ordinarily placed on a laboratory bench/table or inside/next to lab equipment.
   Do NOT place a component on a sink, floor, wall, or unrelated surface unless the
   image visibly shows it there. A sink must never be inferred merely because the
   scene is a laboratory.
5. Distinguish the support surface from the object itself. If a component is on a
   bench/table, say so only when the image provides visual evidence of that surface.
   Do not invent a tabletop, shelf, tray, or container.
6. For an electronic device, explain what it appears to be and what it is visibly
   used for only when the function can be reasonably inferred from its form/context.
7. Read visible labels, markings, numbers or screen text when legible, but never invent
   text that cannot be seen.
8. Mention color, shape, approximate position, connections/wiring and important visible
   details when they genuinely help identify the item.
9. Treat OCR-like text, filenames, user notes, and profile names as NON-VISUAL metadata:
   they must not be copied into the AI description or used to claim that an object is
   present. The uploaded pixels are the sole source of visual claims.
10. Do NOT claim an exact model, part number, medical diagnosis, electrical rating,
    or function when the image does not provide enough evidence. Say "appears to be"
    or "not clearly identifiable" when appropriate.
11. Do not mention that you are an AI and do not output analysis steps.

Return 1-2 informative paragraphs, followed by a concise "Visible technical/details:"
line when technical components or distinctive objects are present.
""".strip()
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            inputs = qwen_processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
        except Exception:
            # Some processor versions need vision information handled internally by
            # the chat template path; retry with tokenize=True.
            inputs = qwen_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
        device = getattr(qwen_model, "device", None)
        if device is not None:
            try:
                inputs = inputs.to(device)
            except Exception:
                pass
        with torch.no_grad():
            output_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=260,
                do_sample=False,
                repetition_penalty=1.08
            )
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, output_ids)]
            text_out = qwen_processor.batch_decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
        else:
            text_out = qwen_processor.batch_decode(
                output_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
        text_out = re.sub(r'\s+', ' ', str(text_out or '')).strip()
        return text_out
    except Exception as e:
        print("Qwen visual description warning:", e)
        return ""


def generate_detailed_description(path, detections, user_description=""):
    """Generate the saved memory's AI description from the IMAGE itself.

    The user_description field is deliberately stored separately for retrieval but
    is never displayed as the generated description. Qwen2.5-VL is preferred for
    richer scene/object reasoning; BLIP is a fallback. YOLO/CLIP detections remain
    evidence and are appended only when they were independently verified.
    """
    image = load_image(path)
    verified = verify_yolo_detections(image, detections)
    detections[:] = verified

    scene_caption = qwen_image_description(image)
    if not scene_caption and blip_model is not None and blip_processor is not None:
        try:
            scene_caption = blip_caption(image).strip()
        except Exception as e:
            print("BLIP full-image description warning:", e)
            scene_caption = ""

    if not scene_caption:
        if verified:
            scene_caption = "The image visibly contains " + ", ".join(
                str(d.get('name', 'object')).strip().lower() for d in verified[:8]
            ) + "."
        else:
            scene_caption = "The image was saved successfully, but no automatic visual description could be generated."

    names = []
    details = []
    for d in verified:
        name = str(d.get('name', '')).strip().title()
        if name and name.lower() not in {x.lower() for x in names}:
            names.append(name)
        pos = str(d.get('position', '')).replace('-', ' ').title()
        cap = str(d.get('description', '')).strip()
        if cap:
            details.append(f"📌 {name} — {cap} • {pos}")
        elif name:
            details.append(f"📌 {name} • {pos}")
        if len(details) >= 10:
            break

    parts = ["🧠 AI visual description:\n" + scene_caption]
    if names:
        parts.append("✅ Confirmed visible objects: " + ", ".join(names[:12]) + ".")
    if details:
        parts.append("\n".join(details))
    return "\n\n".join(parts)

def build_user_friendly_evidence(memory):
    """Compact generic evidence; never dumps the complete saved description."""
    if not memory:
        return "🔎 No verified evidence available."
    parts=[]
    if memory.get('area'):
        parts.append(f"📍 {memory['area']}")
    if memory.get('day') or memory.get('date'):
        parts.append(f"🕒 {memory.get('day','')}, {memory.get('date','')} at {memory.get('time','')}")
    names=[]
    for d in memory.get('objects',[]) or []:
        n=str(d.get('name','')).strip()
        if n and n.lower() not in {x.lower() for x in names}:
            names.append(n.title())
    if names:
        parts.append("✅ Confirmed: "+", ".join(names[:8]))
    return "\n".join(parts) if parts else "🧠 Memory analyzed successfully."


def _clip_image_tensor(image_batch):
    """Version-compatible CLIP image feature extraction.
    Some transformers versions return BaseModelOutputWithPooling from get_image_features;
    that object does not support norm(). We explicitly run the vision encoder and projection.
    """
    if clip_model is None or clip_processor is None:
        return None
    inputs = clip_processor(images=image_batch, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    vision_outputs = clip_model.vision_model(pixel_values=pixel_values, return_dict=True)
    pooled = vision_outputs.pooler_output
    features = clip_model.visual_projection(pooled)
    return features


def get_clip_embedding(image):
    if clip_model is None or clip_processor is None:
        return []
    try:
        with torch.no_grad():
            features = _clip_image_tensor([image])
        features = features / features.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return features[0].detach().cpu().tolist()
    except Exception as e:
        print("CLIP warning:", e)
        return []


def text_embedding(text):
    if embedding_model is None:
        return []
    try:
        return embedding_model.encode(text, normalize_embeddings=True).tolist()
    except Exception as e:
        print("Embedding warning:", e)
        return []


# ============================================================
# ============================================================
# PERSONAL OBJECTS / PLACES — PERSISTENT IMAGE + VISUAL PROFILE
# ============================================================

def save_reference_image(file, prefix):
    if not file or not file.filename:
        raise ValueError("Please select a reference image.")
    ext=secure_image_extension(file.filename)
    final_name=prefix+"_reference"+ext
    temp_name=prefix+"_uploading"+ext
    final_path=REFERENCE_DIR/final_name
    temp_path=REFERENCE_DIR/temp_name
    try:
        file.save(str(temp_path))
        if not temp_path.exists() or temp_path.stat().st_size==0:
            raise IOError("Reference image was not saved to disk.")
        with Image.open(temp_path) as test:
            test.verify()
        temp_path.replace(final_path)
        raw=final_path.read_bytes()
        if not raw:
            raise IOError("Reference image is empty.")
        return final_name,final_path,raw
    finally:
        try:
            if temp_path.exists(): temp_path.unlink()
        except Exception: pass


def clip_image_batch(images):
    if clip_model is None or clip_processor is None or not images:
        return []
    try:
        with torch.no_grad():
            features = _clip_image_tensor(images)
        features = features / features.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return features.detach().cpu().tolist()
    except Exception as e:
        print("CLIP batch warning:", e)
        # Safe per-image fallback; one bad image must not destroy the whole memory save.
        result=[]
        for im in images:
            try:
                result.append(get_clip_embedding(im))
            except Exception:
                result.append([])
        return result


def reference_views(image):
    views=[image]
    w,h=image.size
    side=int(min(w,h)*0.80)
    if side>20 and side<min(w,h):
        x=(w-side)//2; y=(h-side)//2
        views.append(image.crop((x,y,x+side,y+side)))
    for rows,cols,ov in ((2,2,.08),(3,3,.10)):
        tw,th=w/cols,h/rows
        for r in range(rows):
            for c in range(cols):
                x1=max(0,int(c*tw-ov*tw)); y1=max(0,int(r*th-ov*th))
                x2=min(w,int((c+1)*tw+ov*tw)); y2=min(h,int((r+1)*th+ov*th))
                views.append(image.crop((x1,y1,x2,y2)))
    return views


def profile_vectors(image):
    return [v for v in clip_image_batch(reference_views(image)) if v]


def save_personal_object(name,owner,location,category,description,file,browser_time=""):
    name=(name or '').strip(); owner=(owner or '').strip(); location=(location or '').strip(); category=(category or '').strip(); description=(description or '').strip()
    if not name: raise ValueError("Object name is required.")
    prefix="object_"+uuid.uuid4().hex
    filename,path,raw=save_reference_image(file,prefix)
    image=load_image(path)
    try: vectors=profile_vectors(image)
    except Exception as e: print("Object visual profile warning:",e); vectors=[]
    saved_at=parse_client_datetime(browser_time).isoformat(timespec='seconds')
    with DB_LOCK:
        conn=db()
        conn.execute("""INSERT INTO personal_objects
            (id,name,owner,location,category,description,reference_image,clip_embedding_json,reference_image_blob,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",(prefix,name,owner,location,category,description,filename,json_dump(vectors),sqlite3.Binary(raw),saved_at))
        conn.commit(); conn.close()
    return prefix


def save_personal_place(name,owner,room,category,description,file,browser_time=""):
    name=(name or '').strip(); owner=(owner or '').strip(); room=(room or '').strip(); category=(category or '').strip(); description=(description or '').strip()
    if not name: raise ValueError("Place name is required.")
    prefix="place_"+uuid.uuid4().hex
    filename,path,raw=save_reference_image(file,prefix)
    image=load_image(path)
    try: vectors=profile_vectors(image)
    except Exception as e: print("Place visual profile warning:",e); vectors=[]
    saved_at=parse_client_datetime(browser_time).isoformat(timespec='seconds')
    with DB_LOCK:
        conn=db()
        conn.execute("""INSERT INTO personal_places
            (id,name,owner,room,category,description,reference_image,clip_embedding_json,reference_image_blob,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",(prefix,name,owner,room,category,description,filename,json_dump(vectors),sqlite3.Binary(raw),saved_at))
        conn.commit(); conn.close()
    return prefix


def get_profile_vectors(raw):
    data=json_load(raw,[])
    if isinstance(data,dict): data=data.get('vectors',[]) or data.get('prototype',[])
    if isinstance(data,list) and data and isinstance(data[0],(int,float)): return [data]
    return [x for x in data if isinstance(x,list)] if isinstance(data,list) else []


def reference_bytes_from_row(row):
    try:
        if row['reference_image_blob']:
            return bytes(row['reference_image_blob'])
    except Exception: pass
    path=REFERENCE_DIR/Path(row['reference_image']).name
    try:
        return path.read_bytes() if path.exists() else b''
    except Exception: return b''


def image_regions(image,detections=None):
    """Generate object/context crops plus dense grid regions.
    Every visual region carries a real image bounding box so a matched personal
    object can have spatially meaningful nearby-object evidence."""
    out=[]
    for d in (detections or [])[:100]:
        try:
            out.append((crop_object(image,d['bbox'],.22),d))
            out.append((crop_object(image,d['bbox'],.50),d))
        except Exception: pass
    w,h=image.size
    for rows,cols,ov in ((2,2,.10),(3,3,.14),(4,4,.16)):
        tw,th=w/cols,h/rows
        for r in range(rows):
            for c in range(cols):
                x1=max(0,int(c*tw-ov*tw)); y1=max(0,int(r*th-ov*th))
                x2=min(w,int((c+1)*tw+ov*tw)); y2=min(h,int((r+1)*th+ov*th))
                meta={'bbox':[x1,y1,x2,y2],'name':'visual-region'}
                out.append((image.crop((x1,y1,x2,y2)),meta))
    out.append((image,{'bbox':[0,0,w,h],'name':'full-image'}))
    return out


def _dense_visual_candidates(image):
    """Build target-search windows for small/fine-grained personal objects.

    Unlike generic detection, this search has NO class labels and is run only
    for a registered personal profile. Multi-scale overlapping windows make a
    small phone/pen/wallet retrievable even when YOLO does not propose a box.
    """
    W, H = image.size
    candidates = []
    seen = set()

    def add_box(x1, y1, x2, y2):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 <= x1 or y2 <= y1:
            return
        if x2 - x1 < 40 or y2 - y1 < 40:
            return
        key = (x1, y1, x2, y2)
        if key not in seen:
            seen.add(key)
            candidates.append((image.crop((x1, y1, x2, y2)), {"bbox": [x1, y1, x2, y2]}))

    # Broad context windows first. These are inexpensive and often enough for
    # objects that occupy a visible portion of the frame.
    for rows, cols, overlap in ((3, 3, 0.14), (4, 4, 0.16), (5, 5, 0.18)):
        tw, th = W / cols, H / rows
        for r in range(rows):
            for c in range(cols):
                add_box(
                    c * tw - tw * overlap,
                    r * th - th * overlap,
                    (c + 1) * tw + tw * overlap,
                    (r + 1) * th + th * overlap,
                )

    # Extra small-object windows. These are critical when a phone is only a
    # few percent of a landscape frame.
    for frac_w, frac_h, stride in ((0.28, 0.28, 0.20), (0.22, 0.22, 0.17), (0.16, 0.20, 0.13)):
        ww, hh = max(48, int(W * frac_w)), max(48, int(H * frac_h))
        sx, sy = max(24, int(W * stride)), max(24, int(H * stride))
        if ww >= W: xs = [0]
        else: xs = list(range(0, max(1, W - ww + 1), sx))
        if hh >= H: ys = [0]
        else: ys = list(range(0, max(1, H - hh + 1), sy))
        if xs and xs[-1] != max(0, W - ww): xs.append(max(0, W - ww))
        if ys and ys[-1] != max(0, H - hh): ys.append(max(0, H - hh))
        for y in ys:
            for x in xs:
                add_box(x, y, x + ww, y + hh)

    return candidates


def _score_profile_candidates(candidates, ref):
    if not candidates or not ref:
        return []
    try:
        embs = clip_image_batch([x[0] for x in candidates])
    except Exception as e:
        print('Target visual batch warning:', e)
        embs = []
    scored = []
    for (crop, meta), emb in zip(candidates, embs):
        if not emb:
            continue
        score = max(cosine(emb, r) for r in ref)
        scored.append((float(score), meta.get('bbox', [])))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _consensus_dense_match(scored):
    """Return the strongest dense candidate plus a small amount of local support.

    This is deliberately a ranking helper, not the final identity decision.  A
    single hard consensus threshold was causing genuine small-object sightings to
    disappear before the identity verifier ever got a chance to inspect them.
    """
    if not scored:
        return -1.0, None, 0
    best, best_bbox = scored[0]
    if len(best_bbox) != 4:
        return -1.0, None, 0
    supports = 0
    for score, bbox in scored[1:60]:
        if len(bbox) != 4:
            continue
        if score >= best - 0.07 and iou(best_bbox, bbox) >= 0.05:
            supports += 1
    return float(best), best_bbox, supports


def _target_visual_candidates(image, row, detections=None):
    """Build one clean ranked candidate set for a registered target.

    Detector crops are useful only when their detector class is compatible with
    the profile.  Regardless of YOLO, dense target windows are always searched
    because fine-grained items such as a particular blue pen are not COCO classes.
    """
    ref = ensure_profile_vectors(row, 'object' if str(row['id']).startswith('object_') else 'place')
    if not ref or clip_model is None:
        return []

    candidates=[]
    detections=detections or []
    expected=set(profile_expected_classes(row)) if 'profile_expected_classes' in globals() else set()

    # Detector regions: only verified detections; incompatible classes are ignored.
    for d in detections[:100]:
        if d.get('verification') != 'verified':
            continue
        label=norm_text(d.get('name',''))
        if expected and label not in expected:
            continue
        try:
            crop=crop_object(image,d.get('bbox',[]),0.18)
            if crop is not None:
                candidates.append((crop, d.get('bbox',[]), 'detector', label, float(d.get('confidence',0) or 0)))
        except Exception:
            pass

    # Always include dense windows.  These contain no label assumption.
    for crop,meta in _dense_visual_candidates(image):
        candidates.append((crop,meta.get('bbox',[]),'dense-target','',0.0))

    # Full image is useful as weak context, but is never accepted as the identity crop
    # by itself for a personal object.
    if row['id'].startswith('place_'):
        candidates.append((image,[0,0,image.size[0],image.size[1]],'full-image','',0.0))

    if not candidates:
        return []

    embs=clip_image_batch([x[0] for x in candidates])
    ranked=[]
    for item,emb in zip(candidates,embs):
        if not emb:
            continue
        sc=max(cosine(emb,r) for r in ref)
        ranked.append({
            'score':float(sc), 'bbox':item[1], 'source':item[2],
            'label':item[3], 'det_conf':item[4], 'crop':item[0]
        })
    ranked.sort(key=lambda x:x['score'],reverse=True)
    return ranked[:80]


def best_visual_match(image,row,detections=None):
    """Find the best actual visual region for one registered profile.

    Identity is tied to the saved reference image, never to room/location or to
    another memory's metadata.  The function intentionally returns the strongest
    candidate instead of throwing away a real candidate with an overly strict
    dense-window consensus rule.  ``exact_target_visual_match`` performs the final
    identity acceptance gate.
    """
    ranked=_target_visual_candidates(image,row,detections)
    if not ranked:
        return -1.0,None,'none'

    best=ranked[0]
    # Prefer a detector crop only when it is semantically compatible and clearly
    # competitive; otherwise the dense target crop is the identity candidate.
    if best['source']=='detector':
        return best['score'],best['bbox'],'detector'
    return best['score'],best['bbox'],'dense-target'

def match_personal_objects(image,detections):
    """Attach a memory to a registered object only after a high-confidence identity check.

    IMPORTANT:
    - A generic CLIP resemblance is NOT enough to claim that a saved image contains
      a particular user's object.
    - When Qwen is available, the top visual candidate must receive an explicit
      MATCH from the pairwise reference-vs-memory verifier.
    - When Qwen is unavailable, only an exceptionally strong CLIP match with local
      consensus is accepted.
    This deliberately favors NO MATCH over a false wallet/pen/charger sighting.
    """
    with DB_LOCK:
        conn=db()
        rows=conn.execute('SELECT * FROM personal_objects ORDER BY created_at ASC').fetchall()
        conn.close()

    out=[]
    for row in rows:
        try:
            ranked=_target_visual_candidates(image,row,detections)
        except Exception as e:
            print("Personal-object matching warning:",e)
            continue
        if not ranked:
            continue

        # Never use a full-frame embedding as personal-object evidence.
        ranked=[x for x in ranked if len(x.get('bbox',[]))==4 and x.get('source')!='full-image']
        if not ranked:
            continue

        top=ranked[:6]
        best=top[0]
        best_score=float(best.get('score',-1.0))

        confirmed=False
        evidence_source=''
        confirmed_bbox=best.get('bbox') or []

        # PRIMARY PATH: explicit pairwise identity verification.
        if qwen_model is not None and qwen_processor is not None:
            for cand in top:
                score=float(cand.get('score',-1.0))
                if score < 0.64:
                    continue
                try:
                    verified,qconf,qsource=qwen_verify_same_registered_object(
                        image,row,cand.get('crop')
                    )
                except Exception as e:
                    print("Qwen personal-object verification warning:",e)
                    verified,qconf,qsource=False,0.0,'qwen-error'

                if qsource == 'MATCH' and verified and float(qconf) >= 0.70:
                    confirmed=True
                    confirmed_bbox=cand.get('bbox') or []
                    evidence_source='qwen-verified'
                    best_score=min(0.999, score*0.75 + float(qconf)*0.25)
                    break

                # Explicit NO_MATCH is a hard rejection for this candidate.
                if qsource == 'NO_MATCH':
                    continue

            # Qwen present but could not explicitly confirm identity:
            # do NOT silently downgrade to a generic CLIP match.
            if not confirmed:
                continue

        else:
            # SAFE OFFLINE/NO-QWEN FALLBACK:
            # require both an exceptionally high visual score and spatial consensus.
            supports=sum(
                1 for cand in ranked[1:40]
                if float(cand.get('score',-1.0)) >= best_score-0.045
                and iou(best.get('bbox',[]),cand.get('bbox',[])) >= 0.05
            )
            if best_score >= 0.88 and (supports >= 1 or best_score >= 0.91):
                confirmed=True
                evidence_source='visual-consensus'
                confirmed_bbox=best.get('bbox') or []
            else:
                continue

        if confirmed:
            out.append({
                'id':row['id'],
                'personal_name':row['name'],
                'owner':row['owner'],
                'location':row['location'] if 'location' in row.keys() else '',
                'category':row['category'],
                'description':row['description'],
                'similarity':round(float(best_score),4),
                'evidence_bbox':confirmed_bbox,
                'evidence_source':evidence_source
            })

    # Only strong identities are persisted; generic nearby similarity never becomes
    # a personal-object record.
    out.sort(key=lambda x:x['similarity'],reverse=True)
    return out[:20]

def match_personal_place(image):
    """Return only a strong registered place match."""
    with DB_LOCK:
        conn=db()
        rows=conn.execute('SELECT * FROM personal_places ORDER BY created_at ASC').fetchall()
        conn.close()

    best=None
    best_score=-1.0
    for row in rows:
        try:
            score,_,source=best_visual_match(image,row,[])
        except Exception as e:
            print("Personal-place matching warning:",e)
            continue
        threshold=0.55
        if score>best_score:
            best_score=score
            best=row

    if best is not None and best_score>=threshold:
        return {
            'id':best['id'],
            'name':best['name'],
            'owner':best['owner'],
            'room':best['room'],
            'category':best['category'],
            'description':best['description'],
            'similarity':round(best_score,4)
        }
    return None


def list_personal_objects():
    with DB_LOCK:
        conn=db(); rows=conn.execute('SELECT id,name,owner,location,category,description,reference_image,created_at FROM personal_objects ORDER BY created_at DESC').fetchall(); conn.close()
    return [dict(r) for r in rows]


def list_personal_places():
    with DB_LOCK:
        conn=db(); rows=conn.execute('SELECT id,name,owner,room,category,description,reference_image,created_at FROM personal_places ORDER BY created_at DESC').fetchall(); conn.close()
    return [dict(r) for r in rows]
def sync_reference_blobs():
    """Backfill the SQLite image BLOB for older profiles using their existing reference file."""
    with DB_LOCK:
        conn=db()
        try:
            for table in ('personal_objects','personal_places'):
                rows=conn.execute(f"SELECT id,reference_image,reference_image_blob FROM {table}").fetchall()
                for row in rows:
                    if row['reference_image_blob']:
                        continue
                    ref=REFERENCE_DIR/Path(row['reference_image']).name
                    if ref.exists():
                        try:
                            conn.execute(f"UPDATE {table} SET reference_image_blob=? WHERE id=?",(sqlite3.Binary(ref.read_bytes()),row['id']))
                        except Exception as e:
                            print(f"{table} reference BLOB sync warning:",e)
            conn.commit()
        finally:
            conn.close()


# ============================================================
# SPATIAL REASONING
# ============================================================

def build_spatial_context(detections):
    relationships = []
    for a in detections[:80]:
        ax, ay = a["center"]
        for b in detections[:80]:
            if a is b or a["name"] == b["name"] and a["bbox"] == b["bbox"]:
                continue
            bx, by = b["center"]
            dx, dy = bx - ax, by - ay
            dist = math.sqrt(dx * dx + dy * dy)
            rel = []
            if abs(dx) > 0.12 * max(1, abs(ax) + abs(bx)):
                rel.append("left of" if dx > 0 else "right of")
            if abs(dy) > 0.12 * max(1, abs(ay) + abs(by)):
                rel.append("above" if dy > 0 else "below")
            if dist < 220:
                rel.append("near")
            if rel:
                relationships.append({"object": a["name"], "relation": rel, "reference": b["name"], "distance": round(dist, 1)})
    return relationships[:300]


def generate_tags(detections, personal_matches, place, relationships, area):
    tags = set()
    for d in detections:
        tags.add(d["name"])
        tags.add(d["position"])
    for p in personal_matches:
        tags.add(p["personal_name"])
        if p.get("owner"): tags.add(p["owner"])
        if p.get("location"): tags.add(p["location"])
    if place:
        tags.add(place["name"])
        if place.get("room"): tags.add(place["room"])
    for r in relationships:
        tags.add(r["object"]); tags.add(r["reference"])
        tags.update(r["relation"])
    if area and area != "Location unavailable":
        tags.add(area[:80])
    return sorted(t for t in tags if t)


# ============================================================
# PERSONAL OBJECT NEARBY CONTEXT
# ============================================================

def bbox_center(bbox):
    if not bbox or len(bbox) != 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def description_object_hints(text):
    # User descriptions are searchable memory annotations, never automatic nearby-object detections.
    return []


def enrich_nearby_with_user_description(nearby, user_description):
    # Deliberately do NOT convert user-written prose into detected/nearby objects.
    # This prevents descriptions from contaminating visual evidence or retrieval.
    return list(nearby or [])[:18]


def filter_nearby_context_objects(items, area_text='', semantic_place='', target_location=''):
    """Remove semantically implausible nearby labels from target evidence.

    Nearby evidence must come from the exact saved image, be spatially close, and
    make sense in the recorded scene. Labels such as toilet/tub/sink are never
    reported around an office/classroom/study target unless the scene explicitly
    indicates a restroom/bathroom context.
    """
    context=norm_text(' '.join([str(area_text or ''),str(semantic_place or ''),str(target_location or '')]))
    restroom_words={'toilet','bathroom','restroom','washroom','lavatory','shower'}
    office_words={'office','lab','laboratory','classroom','lecture','faculty','work','study','desk','computer'}
    forbidden_general={'tv','toilet','sink','microwave','oven','refrigerator','bathtub','shower','toothbrush'}
    allowed=[]
    for item in items or []:
        name=norm_text(item.get('object',''))
        if not name:
            continue
        # These classes are especially prone to false positives in desk/lab scenes.
        # Never report them as nearby unless the recorded context explicitly supports them.
        if name in {'toilet','sink','microwave','oven','refrigerator','bathtub','shower','toothbrush'}:
            if not any(w in context for w in restroom_words):
                continue
        if name=='tv':
            # Require explicit TV/television evidence in the target context.
            if not any(w in context for w in {'tv','television','television room','media room','living room','lounge'}):
                continue
            if any(w in context for w in office_words) and not any(w in context for w in {'tv','television','media room','living room','lounge'}):
                continue
        allowed.append(item)
    return allowed


def nearby_objects_for_personal_matches(personal_matches, detections, max_items=2, image_size=None):
    """Return only strong, visibly supported neighbors from THIS exact image.
    No caption-only, grid-only, user-described, or low-confidence candidate is used.
    """
    if not personal_matches or not detections:
        return []

    verified=[]
    for d in detections:
        conf=safe_float(d.get('confidence'))
        center=bbox_center(d.get('bbox'))
        if not d.get('name') or conf is None or conf < 0.72 or center is None:
            continue
        if d.get('verification') != 'verified':
            continue
        verified.append((d, center))

    if not verified:
        return []

    if image_size:
        W,H=float(image_size[0]),float(image_size[1])
    else:
        coords=[d.get('bbox',[]) for d,_ in verified if len(d.get('bbox',[]))==4]
        if not coords:
            return []
        W=max(float(b[2]) for b in coords)
        H=max(float(b[3]) for b in coords)
    diag=max(1.0, math.sqrt(W*W+H*H))
    max_dist=max(28.0, diag*0.045)

    candidates=[]
    for pm in personal_matches[:3]:
        pcenter=bbox_center(pm.get('evidence_bbox'))
        pbbox=pm.get('evidence_bbox') or []
        if pcenter is None:
            continue
        for d,center in verified:
            bbox=d.get('bbox') or []
            if len(pbbox)==4 and len(bbox)==4 and iou(pbbox,bbox)>=0.10:
                continue
            dist=math.dist(center,pcenter)
            if dist <= max_dist:
                candidates.append({
                    'object': str(d.get('name','')).strip(),
                    'confidence': d.get('confidence',0),
                    'position': d.get('position',''),
                    'bbox': bbox,
                    'distance_pixels': round(dist,1),
                    'near_personal_object': pm.get('personal_name',''),
                    'verification': 'verified'
                })

    unique=[]; seen=set()
    for x in sorted(candidates, key=lambda z:(z['distance_pixels'],-float(z.get('confidence',0) or 0))):
        key=(x['object'],tuple(x.get('bbox',[])))
        if key in seen:
            continue
        seen.add(key); unique.append(x)
        if len(unique)>=max_items:
            break
    return filter_nearby_context_objects(unique)


# ============================================================
# MEMORY SAVE / LOAD
# ============================================================


def create_memory_text(timestamp, area, detections, personal_matches, place, relationships, description, tags, nearby_objects=None):
    """Create searchable text without mixing one personal object's data with another."""
    parts = ["EchoMind episodic memory", f"timestamp {timestamp}", f"area {area}", description or ""]
    parts.append("objects " + ", ".join(str(d.get("name", "")) for d in (detections or [])))
    parts.append("personal objects " + ", ".join(
        f"{p.get('personal_name','')} {p.get('owner','')} {p.get('location','')} {p.get('category','')}"
        for p in (personal_matches or [])
    ))
    if place:
        parts.append("personal place " + " ".join(str(place.get(k, "")) for k in ("name", "owner", "room", "category")))
    parts.append("spatial " + ". ".join(
        f"{r.get('object','')} {' '.join(r.get('relation', []))} {r.get('reference','')}"
        for r in (relationships or [])
    ))
    if nearby_objects:
        parts.append("nearby objects " + ", ".join(
            str(n.get("object", "")) for n in nearby_objects if n.get("object")
        ))
    parts.append("tags " + ", ".join(str(t) for t in (tags or [])))
    return "\n".join(parts)


def save_memory(image_filename, original_filename, timestamp, original_exif_time, latitude, longitude, gps_accuracy, location_source, area, detections, personal_matches, place, relationships, description, tags, user_description="", clip_regions=None, nearby_objects=None, capture_time_source="Unknown"):
    """Persist a memory using an explicit column/value dictionary.
    This avoids every historical positional INSERT mismatch (including the old
    30-values-for-29-columns failure) while preserving the existing schema."""
    memory_id = "memory_" + uuid.uuid4().hex
    dt = parse_client_datetime(timestamp)
    search_text = create_memory_text(
        timestamp, area, detections, personal_matches, place,
        relationships, description, tags, nearby_objects
    )
    emb = text_embedding(search_text)

    # Date / Day / Time shown by EchoMind are the save/upload time.
    # Original EXIF remains stored separately and never replaces this timestamp.
    date_value = dt.strftime("%Y-%m-%d")
    day_value = dt.strftime("%A")
    time_value = dt.strftime("%H:%M:%S")

    payload = {
        "id": memory_id,
        "image": image_filename,
        "image_url": "/image/" + image_filename,
        "original_filename": original_filename or "",
        "capture_timestamp": timestamp,
        "capture_time_source": capture_time_source or "Unknown",
        "date": date_value,
        "day": day_value,
        "time": time_value,
        "original_exif_time": original_exif_time or "",
        "latitude": latitude,
        "longitude": longitude,
        "gps_accuracy": gps_accuracy,
        "location_source": location_source or "Unavailable",
        "area": area or "Location unavailable",
        "semantic_place": place.get("name", "") if place else "",
        "scene": description.split("\n\nObject-level evidence:")[0] if description else "",
        "description": description or "",
        "user_description": user_description or "",
        "clip_regions_json": json_dump(clip_regions or []),
        "objects_json": json_dump(detections or []),
        "personal_objects_json": json_dump(personal_matches or []),
        "nearby_objects_json": json_dump(nearby_objects or []),
        "place_json": json_dump(place or {}),
        "relationships_json": json_dump(relationships or []),
        "tags_json": json_dump(tags or []),
        "status": "saved",
        "search_text": search_text,
        "embedding_json": json_dump(emb),
        "created_at": dt.isoformat(timespec="seconds"),
    }

    required_columns = [
        "id","image","image_url","original_filename","capture_timestamp","capture_time_source","date","day","time",
        "original_exif_time","latitude","longitude","gps_accuracy","location_source","area",
        "semantic_place","scene","description","user_description","clip_regions_json","objects_json",
        "personal_objects_json","nearby_objects_json","place_json","relationships_json","tags_json",
        "status","search_text","embedding_json","created_at"
    ]

    with DB_LOCK:
        conn = db()
        try:
            actual = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
            missing = [c for c in required_columns if c not in actual]
            if missing:
                raise RuntimeError("Memory database is missing columns: " + ", ".join(missing))
            cols = ",".join(required_columns)
            placeholders = ",".join("?" for _ in required_columns)
            values = [payload[c] for c in required_columns]
            conn.execute(f"INSERT INTO memories ({cols}) VALUES ({placeholders})", values)
            conn.commit()
        finally:
            conn.close()

    return get_memory(memory_id)

def get_memory(memory_id):
    with DB_LOCK:
        conn = db(); row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone(); conn.close()
    return row_to_memory(row) if row else None


def row_to_memory(row):
    if row is None: return None
    d = dict(row)
    d["clip_regions"] = json_load(d.pop("clip_regions_json", "[]"), [])
    d["nearby_objects"] = json_load(d.pop("nearby_objects_json", "[]"), [])
    d["objects"] = json_load(d.pop("objects_json", "[]"), [])
    d["personal_objects"] = json_load(d.pop("personal_objects_json", "[]"), [])
    d["place"] = json_load(d.pop("place_json", "{}"), {})
    d["relationships"] = json_load(d.pop("relationships_json", "[]"), [])
    d["tags"] = json_load(d.pop("tags_json", "[]"), [])
    d["embedding"] = json_load(d.pop("embedding_json", "[]"), [])
    return d


def all_memories():
    with DB_LOCK:
        conn = db(); rows = conn.execute("SELECT * FROM memories ORDER BY created_at DESC").fetchall(); conn.close()
    return [row_to_memory(r) for r in rows]


# ============================================================
# ============================================================
# RETRIEVAL / CHATBOT — PERSONAL PROFILES FIRST
# ============================================================
STOPWORDS={'where','is','my','the','a','an','did','i','see','last','latest','when','what','was','this','that','in','on','at','of','to','please','can','you','tell','me','find','leave','left','seen','saw','show','give','for','were','it','there','do','does','has','have','had','with','from','about','are','located','did'}

def norm_text(x):
    x=(x or '').lower().replace('’',"'")
    return re.sub(r'\s+',' ',re.sub(r"[^a-z0-9']+",' ',x)).strip()

def query_terms(q):
    return [w.strip("'") for w in re.findall(r"[a-zA-Z0-9']+",(q or '').lower()) if w not in STOPWORDS and len(w)>1]

def phrase_score(query,text):
    qn=norm_text(query); tn=norm_text(text)
    if not qn or not tn: return 0.0
    if qn in tn: return 1.0
    qt=query_terms(qn); tt=set(query_terms(tn))
    return (sum(1 for t in qt if t in tt)/len(qt)) if qt else 0.0

def profile_match_score(q,row,is_place=False):
    """High-precision query-to-profile matching.

    Exact profile names win. Metadata supports alternate wording such as
    'Isra blue pen' when the registered profile stores owner/category/description.
    """
    qn=norm_text(q)
    if not qn:
        return 0.0
    name=norm_text(row['name'])
    score=0.0
    if name and name == qn:
        return 100000.0
    if name and name in qn:
        score += 50000.0 + len(name)*100.0

    qtokens=set(query_terms(qn))
    profile_fields=[
        str(row['name'] or ''),
        str(row['owner'] or ''),
        str(row['category'] or ''),
        str(row['description'] or '')
    ]
    if is_place:
        profile_fields.append(str(row['room'] or ''))
    ptokens=set(query_terms(' '.join(profile_fields)))

    # Require meaningful token overlap. Generic question words do not influence target choice.
    overlap=qtokens & ptokens
    score += len(overlap)*500.0

    # Reward adjacent exact phrases from the profile's name/description.
    for field in profile_fields:
        fn=norm_text(field)
        if fn and fn in qn:
            score += min(5000.0, len(fn)*80.0)

    return score


def registered_targets(query):
    with DB_LOCK:
        conn=db()
        objs=conn.execute('SELECT * FROM personal_objects').fetchall()
        places=conn.execute('SELECT * FROM personal_places').fetchall()
        conn.close()
    out=[]
    for r in objs:
        sc=profile_match_score(query,r,False)
        if sc>0:
            out.append((sc,'object',r))
    for r in places:
        sc=profile_match_score(query,r,True)
        if sc>0:
            out.append((sc,'place',r))
    out.sort(key=lambda x:(x[0],-(len(str(x[2]['name'] or '')))),reverse=True)
    return out


def select_target_profile(query):
    candidates=registered_targets(query)
    if not candidates:
        return None

    # Exact full-name query is always deterministic.
    qn=norm_text(query)
    exact=[c for c in candidates if norm_text(c[2]['name'])==qn]
    if exact:
        exact.sort(key=lambda x:x[2]['created_at'] or '')
        return exact[0]

    # If multiple profiles have only a generic shared term ('pen'), do not silently
    # mix them. Prefer a uniquely stronger candidate; otherwise let retrieval aggregate
    # all matching profiles explicitly.
    candidates.sort(key=lambda x:x[0],reverse=True)
    if len(candidates)==1:
        return candidates[0]
    if candidates[0][0] >= candidates[1][0]*2.0:
        return candidates[0]
    return candidates[0]


def memory_blob(m):
    # The handwritten note is PRIVATE to one exact memory record. Never use it
    # for cross-memory text/semantic matching, otherwise a note such as
    # "resistor, potentiometer on the lab table" can pull an unrelated image
    # into a query and make its note appear to belong to that image.
    parts=[m.get('search_text',''),m.get('description',''),
           m.get('scene',''),m.get('area','')]
    parts += [str(x.get(k,'')) for x in m.get('personal_objects',[])
              for k in ('id','personal_name','owner','location','category','description')]
    pl=m.get('place',{}) or {}
    parts += [str(pl.get(k,'')) for k in ('id','name','owner','room','category','description')]
    parts += [str(x.get('name','')) for x in m.get('objects',[])]
    parts += list(m.get('tags',[]))
    # nearby objects are evidence for this memory only
    parts += [str(x.get('object','')) for x in m.get('nearby_objects',[])]
    return norm_text(' '.join(parts))


def ensure_profile_vectors(row, kind):
    vectors=get_profile_vectors(row["clip_embedding_json"])
    if vectors:
        return vectors
    raw=reference_bytes_from_row(row)
    if not raw or clip_model is None:
        return []
    try:
        vectors=profile_vectors(Image.open(io.BytesIO(raw)).convert('RGB'))
        if vectors:
            table='personal_objects' if kind=='object' else 'personal_places'
            with DB_LOCK:
                conn=db()
                conn.execute(f"UPDATE {table} SET clip_embedding_json=? WHERE id=?",
                             (json_dump(vectors),row["id"]))
                conn.commit()
                conn.close()
        return vectors
    except Exception as e:
        print("Profile vector repair warning:",e)
        return []


def profile_expected_classes(row):
    """Infer conservative detector classes from a registered profile.
    Used only as an additional guard; CLIP still decides identity for classes like pen."""
    text=norm_text(' '.join(str(row[k] or '') for k in ('name','category','description') if k in row.keys()))
    mapping={
        'bottle':'bottle','water bottle':'bottle','cup':'cup','mug':'cup','keyboard':'keyboard',
        'mouse':'mouse','laptop':'laptop','phone':'cell phone','smartphone':'cell phone',
        'charger':'cell phone','backpack':'backpack','bag':'backpack','handbag':'handbag',
        'suitcase':'suitcase','remote':'remote','book':'book','notebook':'book','chair':'chair',
        'clock':'clock','plant':'potted plant','scissors':'scissors'
    }
    found=[]
    for phrase,cls in sorted(mapping.items(), key=lambda x:len(x[0]), reverse=True):
        if phrase in text and cls not in found:
            found.append(cls)
    return found


def cached_visual_score(memory,row):
    """Match a profile only against appropriate, stored detector regions.
    For known object classes (e.g. bottle), unrelated classes are excluded before
    CLIP scoring. For fine-grained classes absent from YOLO (e.g. pen), CLIP alone
    may still be used, but at a higher threshold in exact_target_visual_match()."""
    vectors=ensure_profile_vectors(row, 'object' if str(row['id']).startswith('object_') else 'place')
    if not vectors:
        return -1.0,None,"none"

    expected=profile_expected_classes(row)
    best=-1.0; bbox=None; source='none'
    for region in memory.get('clip_regions',[]) or []:
        emb=region.get('embedding',[])
        if not emb or str(row['id']).startswith('object_') and region.get('source') not in {'YOLO-region','detector'}:
            continue
        # The bbox links the cached embedding back to the detector record.
        rbbox=region.get('bbox') or []
        if expected and len(rbbox)==4:
            matching_detection=False
            for d in memory.get('objects',[]) or []:
                if d.get('bbox') and iou(d.get('bbox'),rbbox)>=0.30 and norm_text(d.get('name','')) in expected and d.get('verification')=='verified':
                    matching_detection=True
                    break
            if not matching_detection:
                continue
        sc=max(cosine(emb,ref) for ref in vectors)
        if sc>best:
            best=sc; bbox=rbbox; source=str(region.get('source','none'))
    return best,bbox,source


def qwen_verify_same_registered_object(candidate_image, row, target_crop=None):
    """Use the VLM as a final identity gate for a registered personal object.

    CLIP is excellent for retrieval but can confuse semantically similar objects.
    This verifier asks Qwen2.5-VL to compare the saved candidate crop against the
    registered reference image and return a strict MATCH/NO_MATCH decision. It is
    used only for top visual candidates, keeping normal retrieval responsive while
    preventing an unrelated image from being shown as the user's object.
    """
    # Qwen is a final confirmation layer, not a hard dependency. If it is not
    # loaded or cannot run in the current environment, fall back to the verified
    # CLIP/reference-image score instead of hiding every otherwise valid picture.
    if qwen_model is None or qwen_processor is None or not candidate_image or not row:
        return True, 0.0, 'qwen-unavailable'
    raw = reference_bytes_from_row(row)
    if not raw:
        return True, 0.0, 'reference-unavailable'
    try:
        reference = Image.open(io.BytesIO(raw)).convert('RGB')
        candidate = (target_crop or candidate_image).convert('RGB')
    except Exception as e:
        print('Qwen identity image warning:', e)
        return False, 0.0, 'image-error'

    target_name = str(row['name'] or '').strip()
    target_desc = str(row['description'] or '').strip()
    prompt = f"""
You are the final identity verifier for a personal-memory assistant.

REGISTERED PERSONAL OBJECT: {target_name}
REGISTERED DESCRIPTION: {target_desc}

IMAGE 1 is the saved registered reference photo of the exact personal object.
IMAGE 2 is a crop from a saved memory photo.

Decide whether IMAGE 2 contains the SAME PHYSICAL PERSONAL OBJECT as IMAGE 1.
The object may have moved to a different room, surface, lighting condition, or angle.
Ignore the background/location and compare object identity only: shape, dimensions,
color pattern, distinctive markings, logo/label, packaging details, geometry and
other persistent visual characteristics.

Do NOT answer MATCH merely because both images contain the same generic category
(e.g. two bottles, two phones, two boxes, two chargers). A generic category match is
NOT sufficient for a personal-object identity match.
Do NOT use the registered description as proof that the object is present. Use the
pixels in IMAGE 2 and the visual comparison with IMAGE 1.

Return exactly one line in this format:
MATCH|0.00
or
NO_MATCH|0.00
where the number is your confidence from 0 to 1.
""".strip()

    try:
        messages=[{
            'role':'user',
            'content':[
                {'type':'text','text':prompt},
                {'type':'image','image':reference},
                {'type':'image','image':candidate},
            ],
        }]
        try:
            text=qwen_processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=qwen_processor(text=[text],images=[reference,candidate],padding=True,return_tensors='pt')
        except Exception:
            inputs=qwen_processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
        device=getattr(qwen_model,'device',None)
        if device is not None:
            try: inputs=inputs.to(device)
            except Exception: pass
        with torch.no_grad():
            output=qwen_model.generate(**inputs,max_new_tokens=12,do_sample=False)
        ids=inputs.get('input_ids')
        generated=[o[len(i):] for i,o in zip(ids,output)] if ids is not None else output
        reply=qwen_processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=True)[0].strip().upper()
        m=re.search(r'\b(MATCH|NO_MATCH)\s*\|\s*(0(?:\.\d+)?|1(?:\.0+)?)',reply)
        if not m:
            # An unparseable VLM response is not evidence that the object is absent.
            # Let the independent visual matcher decide.
            return True,0.0,'qwen-unparseable'
        decision=m.group(1)
        confidence=float(m.group(2))
        return decision=='MATCH' and confidence>=0.70, confidence, decision
    except Exception as e:
        print('Qwen identity verification warning:',e)
        # Qwen is supplementary. If inference fails, retain the independent CLIP
        # verification path so the working retrieval/UI does not disappear.
        return True,0.0,'qwen-error'

def exact_target_visual_match(memory, row, kind):
    """Verify that THIS stored memory really contains the requested profile.

    For personal objects the saved reference image is the identity source.  The
    memory filename is reopened from disk, candidate regions are ranked against
    that reference, and (when Qwen is available) only the strongest candidates are
    sent to the pairwise identity verifier.  Qwen is a positive confirmation layer:
    an explicit NO_MATCH rejects, while unavailable/error states fall back to the
    independent visual score instead of deleting otherwise valid UI/results.
    """
    if not memory or not row:
        return -1.0,None,'none'
    path=UPLOAD_DIR / Path(str(memory.get('image',''))).name
    if not path.exists() or not path.is_file():
        return -1.0,None,'none'
    try:
        image=load_image(path)
    except Exception as e:
        print(f'{kind.title()} retrieval image warning:',e)
        return -1.0,None,'none'

    try:
        ranked=_target_visual_candidates(image,row,memory.get('objects',[]))
    except Exception as e:
        print(f'{kind.title()} retrieval visual warning:',e)
        return -1.0,None,'none'
    if not ranked:
        return -1.0,None,'none'

    # We need an actual object-sized region for an object.  Never accept the full
    # frame simply because its global CLIP embedding looks similar.
    ranked=[x for x in ranked if len(x.get('bbox',[]))==4 and x.get('source')!='full-image'] if kind=='object' else ranked
    if not ranked:
        return -1.0,None,'none'

    top=ranked[:6]
    top_score=float(top[0]['score'])
    # Basic visual floor.  Qwen can make the final identity decision; this floor
    # prevents obviously unrelated images from being sent to the verifier.
    floor=0.62 if kind=='object' else 0.60
    if top_score < floor:
        return -1.0,None,'none'

    # Pairwise identity verification is the critical guard against wallet/lunchbox/
    # another-pen false positives.  Ask about the top few visual regions, not the
    # whole scene.  Any explicit MATCH is sufficient; explicit NO_MATCH means that
    # candidate is rejected and the next candidate is tested.
    if kind=='object' and qwen_model is not None and qwen_processor is not None:
        for cand in top:
            try:
                crop=cand['crop']
                verified,qconf,qsource=qwen_verify_same_registered_object(image,row,crop)
            except Exception as e:
                print('Qwen target verification warning:',e)
                verified,qconf,qsource=True,0.0,'qwen-error'
            if qsource=='NO_MATCH':
                continue
            if qsource=='MATCH' and verified and float(qconf)>=0.65:
                return min(0.999, float(cand['score'])*0.75 + float(qconf)*0.25), cand['bbox'], 'verified-object'
            if qsource in {'qwen-error','qwen-unparseable','qwen-unavailable','reference-unavailable'}:
                # Do not let a failed auxiliary model erase the visual path.
                if float(cand['score'])>=0.76:
                    return float(cand['score']),cand['bbox'],'visual-fallback'

        return -1.0,None,'none'

    # No Qwen for places, or Qwen not configured: use a stricter independent visual
    # rule.  A strong candidate plus local agreement is enough; otherwise reject.
    if kind=='object':
        best=top[0]
        supports=sum(1 for cand in ranked[1:30]
                     if float(cand['score'])>=float(best['score'])-0.06
                     and iou(best['bbox'],cand['bbox'])>=0.05)
        if float(best['score'])>=0.80 and supports>=1:
            return float(best['score']),best['bbox'],'visual-confirmed'
        if float(best['score'])>=0.86:
            return float(best['score']),best['bbox'],'visual-confirmed'
        return -1.0,None,'none'

    best=top[0]
    if float(best['score'])>=0.70:
        return float(best['score']),best['bbox'],'visual-confirmed'
    return -1.0,None,'none'

def _memory_image_size(memory):
    try:
        path = UPLOAD_DIR / Path(str(memory.get('image', ''))).name
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def memory_time_label(memory):
    """Return one authoritative saved timestamp; never mix date/day/time fields."""
    raw=str(memory.get("capture_timestamp") or memory.get("created_at") or "").strip()
    if raw:
        try:
            dt=datetime.fromisoformat(raw.replace("Z","+00:00"))
            # Stored EchoMind timestamps are already local wall-clock values.
            if dt.tzinfo is not None:
                dt=dt.replace(tzinfo=None)
            return f"{dt.strftime('%A')}, {dt.strftime('%Y-%m-%d')} at {dt.strftime('%H:%M:%S')}"
        except Exception:
            pass
    day=str(memory.get("day") or "").strip()
    date=str(memory.get("date") or "").strip()
    tm=str(memory.get("time") or "").strip()
    if day and date and tm:
        return f"{day}, {date} at {tm}"
    return "Image date/time unavailable"


def populate_retrieval_nearby(memory, bbox):
    """Return only nearby objects that were already verified for this exact memory.

    Retrieval must not run a fresh detector on an old image and then present those
    results as historical evidence. This prevents generic labels from appearing
    beside a target simply because the detector changed between save and retrieval.
    """
    existing=list(memory.get("nearby_objects") or [])
    if not existing:
        return []
    match=memory.get('retrieval_match',{}) or {}
    return filter_nearby_context_objects(
        existing,
        memory.get('area',''),
        memory.get('semantic_place',''),
        match.get('target_location','')
    )[:4]


app.jinja_env.globals["memory_time_label"] = memory_time_label

def user_annotation_for_memory(memory):
    """Return ONLY the handwritten note belonging to this exact memory id.

    SQLite is authoritative. Re-reading by id prevents a retrieved candidate,
    AI description, personal profile, or previous memory from supplying another
    image's note. The note is never used as visual evidence.
    """
    if not memory:
        return ""
    memory_id = str(memory.get("id") or "").strip()
    if memory_id and not memory_id.startswith("profile:"):
        try:
            with DB_LOCK:
                conn = db()
                row = conn.execute(
                    "SELECT user_description FROM memories WHERE id=?",
                    (memory_id,)
                ).fetchone()
                conn.close()
            if row is not None:
                return str(row["user_description"] or "").strip()
        except Exception as e:
            print("User-note isolation warning:", e)
    return str(memory.get("user_description") or "").strip()


def retrieval_saved_note(memory, row=None, kind='object'):
    """Return the user's handwritten description for the registered profile first.

    My Objects / My Places descriptions belong to the profile itself, not to whichever
    memory happened to be retrieved.  This prevents an old capture note from replacing
    the user's saved description for the requested personal item/place.
    """
    try:
        if row is not None:
            note = str(row['description'] or '').strip()
            if note:
                return note
    except Exception:
        pass
    # Safe fallback for legacy memories that did not have a profile description.
    return user_annotation_for_memory(memory)


def build_target_evidence(memory):
    """Very small judge/user-facing evidence panel for the requested target only.
    It deliberately excludes the full scene description and unrelated saved objects."""
    match = memory.get('retrieval_match', {}) or {}
    item = match.get('target_name', 'requested item')
    lines = [f'🎯 {item} found in this memory.']
    saved_location = match.get('target_location') or memory.get('area')
    if saved_location:
        lines.append(f'📍 {saved_location}')
    lines.append(f'🗓️ Captured: {memory_time_label(memory)}')
    if match.get('visual_similarity') is not None:
        lines.append(f'✅ Visual match: {match["visual_similarity"]:.2f}')
    nearby=[]
    for n in memory.get('nearby_objects', []) or []:
        name=str(n.get('object', '')).strip()
        if name and name.lower() not in {x.lower() for x in nearby}:
            nearby.append(name.title())
    if nearby:
        lines.append('👀 Nearby verified: ' + ', '.join(nearby[:4]))
    return '\n'.join(lines)


def _target_text_evidence(memory, row):
    """High-precision textual guard for non-explicit personal-object matches.

    A CLIP-only dense-window match can confuse visually similar unrelated objects
    (for example electronics on a white table) with a fine-grained cosmetic item.
    We therefore require the exact saved memory itself to contain evidence for the
    requested target before a visual-only candidate can become a confirmed sighting.
    Importantly, this guard never uses the My Object profile description as if it
    were evidence from the photographed scene.
    """
    if not memory or not row:
        return False

    # Evidence fields belong to THIS memory only. Do not use personal_objects_json,
    # search_text, or the profile description here because those can be copied or
    # inherited metadata rather than visual evidence from this image.
    evidence_parts = [
        str(memory.get('description') or ''),
        str(memory.get('scene') or ''),
        str(memory.get('area') or ''),
        str(memory.get('original_filename') or ''),
    ]
    evidence_parts += [str(d.get('name') or '') for d in (memory.get('objects') or [])]
    evidence_parts += [str(t or '') for t in (memory.get('tags') or [])]
    evidence = norm_text(' '.join(evidence_parts))
    if not evidence:
        return False

    name = norm_text(row['name'])
    category = norm_text(row['category'])
    profile_desc = norm_text(row['description'])

    # Exact target phrase/name in the captured memory is the strongest textual guard.
    if name and name in evidence:
        return True

    # Build meaningful tokens from the registered profile. Common visual words are
    # deliberately ignored so a generic "bottle"/"box" match cannot confirm a
    # specific named item such as "Rivaj no pore primer".
    ignore = {
        'my','the','a','an','and','with','for','item','object','thing','personal',
        'glass','bottle','box','cover','container','product','photo','picture','image',
        'blue','black','white','red','green','small','large','used','use','in','on','at',
        'owner','person','hers','his','their','room','hand','hands'
    }
    name_tokens = [t for t in query_terms(name) if t not in ignore and len(t) >= 3]
    desc_tokens = [t for t in query_terms(profile_desc) if t not in ignore and len(t) >= 4]
    category_tokens = [t for t in query_terms(category) if t not in ignore and len(t) >= 3]

    # Strong distinguishing words such as "primer", "rivaj", "charger", "wallet"
    # are sufficient when present in the exact captured memory.
    strong = []
    for t in name_tokens + desc_tokens:
        if t not in strong:
            strong.append(t)
    if any(t in evidence.split() for t in strong):
        return True

    # Generic detector classes can still validate a target when the class itself is
    # what was registered (e.g. "phone"), but do not let a generic category validate
    # a highly specific named product.
    class_hits = [t for t in category_tokens if t in evidence.split()]
    if class_hits and not strong:
        return True

    # Exact object-name aliases from the verified detector list are also valid scene
    # evidence (e.g. "cell phone" for a registered phone).
    det_names = {norm_text(d.get('name') or '') for d in (memory.get('objects') or [])}
    det_names.discard('')
    for t in category_tokens:
        if any(t == dn or t in dn or dn in t for dn in det_names):
            return True

    return False


def _target_memory_datetime_parts(memory):
    """Return date/day/time derived from the same timestamp used for the match."""
    raw = str(memory.get('capture_timestamp') or memory.get('created_at') or '').strip()
    try:
        dt = datetime.fromisoformat(raw.replace('Z','+00:00'))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt.strftime('%Y-%m-%d'), dt.strftime('%A'), dt.strftime('%H:%M:%S')
    except Exception:
        return (str(memory.get('date') or ''), str(memory.get('day') or ''), str(memory.get('time') or ''))



def _profile_datetime_parts(profile):
    """Return the registration/save date, day and time of the requested My Object/My Place."""
    try:
        raw = str(profile['created_at'] or '').strip()
    except Exception:
        raw = ''
    if not raw:
        return ('', '', '')
    try:
        dt = parse_client_datetime(raw)
        return dt.strftime('%Y-%m-%d'), dt.strftime('%A'), dt.strftime('%H:%M:%S')
    except Exception:
        try:
            dt = datetime.fromisoformat(raw.replace('Z','+00:00'))
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt.strftime('%Y-%m-%d'), dt.strftime('%A'), dt.strftime('%H:%M:%S')
        except Exception:
            return ('', '', '')


def _memory_has_exact_profile_link(memory, row):
    """Return True only when SQLite metadata explicitly links THIS memory to THIS profile."""
    if not memory or not row:
        return False
    target_id=str(row['id'])
    for p in memory.get('personal_objects',[]) or []:
        # The persistent profile ID is the unique identity key. Do not require a
        # duplicated display name, because older saved records may omit personal_name.
        if str(p.get('id','')).strip()==target_id:
            return True
    return False

def _build_profile_result(row, kind):
    """Build the deterministic PRIMARY RESULT from the registered My Object/My Place record.

    This record is never a historical memory.  Its image, manual description, location,
    and saved timestamp all come directly from the same persistent profile row.
    """
    profile=dict(row)
    pdate,pday,ptime=_profile_datetime_parts(row)
    profile_result={
        'id': 'profile:' + str(row['id']),
        'retrieval_mode': 'target_profile',
        'retrieval_match': {
            'target_type': 'personal_profile',
            'target_id': row['id'],
            'target_name': str(row['name'] or '').strip(),
            'target_owner': str(row['owner'] or '').strip(),
            'target_location': str(row['location'] or '').strip() if kind=='object' and 'location' in row.keys() else '',
            'target_room': str(row['room'] or '').strip() if kind=='place' and 'room' in row.keys() else '',
            'target_category': str(row['category'] or '').strip(),
            'target_description': str(row['description'] or '').strip(),
            'profile_created_at': row['created_at'],
            'profile_saved_label': profile_time_label(row),
            'profile_saved_date': pdate,
            'profile_saved_day': pday,
            'profile_saved_time': ptime,
        },
        'profile_description': str(row['description'] or '').strip(),
        'user_description': str(row['description'] or '').strip(),
        'retrieval_saved_note': str(row['description'] or '').strip(),
        'registered_reference_name': str(row['name'] or '').strip(),
        'registered_reference_description': str(row['description'] or '').strip(),
        'registered_reference_image_url': '/personal-reference/' + ('object' if kind=='object' else 'place') + '/' + str(row['id']),
        'memory_image_url': '/personal-reference/' + ('object' if kind=='object' else 'place') + '/' + str(row['id']),
        'area': str(row['location'] or '').strip() if kind=='object' and 'location' in row.keys() else str(row['room'] or '').strip() if kind=='place' and 'room' in row.keys() else '',
        'date': pdate,
        'day': pday,
        'time': ptime,
        'profile_ai_description': '',
        'retrieval_target_description': '',
        'latitude': None,
        'longitude': None,
    }
    # The reference image is the PRIMARY picture. Generate AI text from that image only.
    try:
        raw=reference_bytes_from_row(row)
        if raw:
            profile_image=Image.open(io.BytesIO(raw)).convert('RGB')
            ai=qwen_image_description(profile_image) if qwen_model is not None and qwen_processor is not None else ''
            if not ai and blip_model is not None and blip_processor is not None:
                ai=blip_caption(profile_image).strip()
            profile_result['profile_ai_description']=str(ai or '').strip()
            profile_result['retrieval_target_description']=profile_result['profile_ai_description']
    except Exception as e:
        print('Profile AI description warning:',e)
    return profile_result


def retrieve_memories(query,limit=5):
    """Deterministic registered-item retrieval.

    PRIMARY RESULT:
      The exact registered My Object/My Place record is ALWAYS first when the query
      resolves to a registered profile. Its image/manual description/save date/day/time/
      location are taken from that profile record itself.

    SECONDARY RESULTS:
      Only saved memories of that same physical object are shown. Explicit SQLite
      profile links are authoritative. Older unlinked memories are accepted only after
      strict pairwise visual identity verification; generic semantic retrieval is never
      allowed to fill the object history.
    """
    memories=all_memories()
    target=select_target_profile(query)

    if not target:
        # Non-profile queries retain the generic memory search, but still bind each
        # image/description/date to the same memory row.
        if not memories:
            return []
        qemb=text_embedding(query)
        terms=query_terms(query)
        scored=[]
        for m in memories:
            blob=memory_blob(m)
            score=sum(10 for t in terms if t in blob)
            if qemb:
                sem=cosine(qemb,m.get('embedding',[]))
                if sem>=0: score+=max(0,sem)*10
            try: score+=datetime.fromisoformat(m['created_at']).timestamp()/1e9
            except Exception: pass
            scored.append((score,m))
        scored.sort(key=lambda x:(x[0],x[1].get('capture_timestamp','')),reverse=True)
        out=[m for _,m in scored[:limit]]
        for m in out:
            m['user_description']=user_annotation_for_memory(m)
            filename=Path(str(m.get('image',''))).name
            m['memory_image_url']='/stored-memory-image/'+str(m.get('id')) if filename and m.get('id') else ''
            m['retrieval_mode']='generic'
            m['user_friendly_evidence']=build_user_friendly_evidence(m)
        return out

    _,kind,row=target
    primary=_build_profile_result(row,kind)
    profile_matches=[]

    # 1) Exact saved Object -> Memory links are authoritative. Do NOT re-run CLIP/Qwen
    # on these rows. A historical visual verifier can fail because of crop/lighting even
    # though the save-time pipeline already explicitly linked this memory to the profile.
    linked=[]
    unlinked=[]
    for m in memories:
        if kind=='object' and _memory_has_exact_profile_link(m,row):
            linked.append(m)
        else:
            unlinked.append(m)

    for m in linked:
        bbox=[]
        visual=0.99
        for pm in m.get('personal_objects',[]) or []:
            if str(pm.get('id',''))==str(row['id']):
                if pm.get('evidence_bbox') and len(pm.get('evidence_bbox'))==4:
                    bbox=pm.get('evidence_bbox')
                try:
                    visual=max(0.90,float(pm.get('similarity') or 0.99))
                except Exception:
                    visual=0.99
                break
        profile_matches.append((m,min(0.999,visual),bbox,'stored-object-link',True))

    # 2) Older memories without a persisted profile link are potentially useful only
    # when they can pass an explicit pairwise identity check. This prevents unrelated
    # objects from appearing just because their text/scene is similar.
    if kind=='object' and qwen_model is not None and qwen_processor is not None:
        for m in unlinked:
            try:
                visual,bbox,evidence_type=exact_target_visual_match(m,row,kind)
            except Exception as e:
                print('Target retrieval warning:',e)
                continue
            if not bbox or evidence_type=='none':
                continue
            if evidence_type=='verified-object' and float(visual)>=0.66:
                profile_matches.append((m,float(visual),bbox,evidence_type,False))
            # When the Qwen path itself becomes unavailable/error, exact_target_visual_match
            # may report visual-fallback, but an unlinked historical memory is NOT allowed
            # to enter the final answer on CLIP alone.

    # Places keep their dedicated visual matching behavior.
    if kind=='place':
        for m in unlinked:
            try:
                visual,bbox,evidence_type=exact_target_visual_match(m,row,kind)
            except Exception:
                continue
            if bbox and evidence_type!='none' and float(visual)>=0.60:
                profile_matches.append((m,float(visual),bbox,evidence_type,False))

    # De-duplicate exact image files, then put persisted linked memories before recovered
    # ones. The profile itself is always outside this list and remains result #1.
    profile_matches.sort(key=lambda x:(1 if x[4] else 0,x[1],x[0].get('capture_timestamp',''),x[0].get('created_at','')),reverse=True)
    unique=[]
    seen_hashes=set()
    for m,visual,bbox,evidence_type,was_linked in profile_matches:
        path=UPLOAD_DIR/Path(str(m.get('image',''))).name
        digest=_file_sha1(path) if path.exists() else ''
        key=('sha1',digest) if digest else ('id',m.get('id'))
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        unique.append((m,visual,bbox,evidence_type,was_linked))

    matched_out=[]
    for m,visual,bbox,evidence_type,was_linked in unique[:max(0,limit-1)]:
        m['retrieval_mode']='target_memory'
        m['matched_memory_timestamp']=memory_time_label(m)
        m['matched_memory_date'],m['matched_memory_day'],m['matched_memory_time']=_target_memory_datetime_parts(m)
        m['retrieval_match']={
            'target_type':kind,'target_id':row['id'],'target_name':row['name'],
            'target_owner':row['owner'],
            'target_location':row['location'] if kind=='object' and 'location' in row.keys() else '',
            'target_room':row['room'] if kind=='place' else '',
            'target_category':row['category'],'target_description':row['description'],
            'profile_created_at':row['created_at'],'profile_saved_label':profile_time_label(row),
            'profile_saved_date':_profile_datetime_parts(row)[0],
            'profile_saved_day':_profile_datetime_parts(row)[1],
            'profile_saved_time':_profile_datetime_parts(row)[2],
            'visual_similarity':round(float(visual),4),'evidence_bbox':bbox,
            'evidence':[f'confirmed visual identity ({float(visual):.2f})',
                        'exact My Object profile link' if was_linked else 'fresh pairwise visual confirmation'],
        }
        if kind=='object':
            m['nearby_objects']=populate_retrieval_nearby(m,bbox)
        else:
            m['nearby_objects']=[]
        # History cards must use THIS memory's own handwritten note, not the profile note.
        m['capture_user_description']=user_annotation_for_memory(m)
        m['user_description']=user_annotation_for_memory(m)
        m['retrieval_saved_note']=m['user_description']
        m['profile_description']=str(row['description'] or '').strip()
        attach_retrieval_image_links(m,row,kind)
        m['retrieval_bound_memory_id']=m.get('id')
        m['retrieval_bound_image_url']=m.get('memory_image_url','')
        m['registered_reference_image_url']=primary.get('registered_reference_image_url','')
        m['registered_reference_name']=primary.get('registered_reference_name','')
        m['registered_reference_description']=primary.get('registered_reference_description','')
        m['retrieval_target_description']=generate_retrieval_target_description(m,row,kind,bbox)
        m['user_friendly_evidence']=build_target_evidence(m)
        matched_out.append(m)

    primary['matched_memory_count']=len(matched_out)
    primary['matched_memories']=matched_out
    primary['history_note']=(
        'No confirmed historical sighting is available for this registered item yet.' if not matched_out else
        'Only one confirmed historical picture is available — no timeline is shown yet.' if len(matched_out)==1 else
        f'{len(matched_out)} confirmed historical pictures of this same item were found.'
    )
    primary['history_images']=[m.get('memory_image_url','') for m in matched_out]
    return [primary]+matched_out

def attach_retrieval_image_links(memory, row, kind):
    """Attach the exact stored image file belonging to this SQLite memory row.

    The memory row already contains the authoritative filename in ``image``.
    Using the normal /image/<filename> route avoids any mismatch between a
    retrieved memory id and an image URL and keeps the database/history/evidence
    views on the same exact stored file.
    """
    if not memory:
        return memory
    filename = Path(str(memory.get("image", ""))).name
    if filename and not str(memory.get("id", "")).startswith("profile:"):
        memory["memory_image_url"] = "/stored-memory-image/" + str(memory.get("id"))
    else:
        memory["memory_image_url"] = ""
    memory["reference_image_url"] = "/personal-reference/" + ("object" if kind == "object" else "place") + "/" + str(row["id"])
    return memory


def find_requested_item(query,memory):
    if memory and memory.get('retrieval_match'):
        return memory['retrieval_match'].get('target_name')
    target=select_target_profile(query)
    return target[2]['name'] if target else None



def generate_retrieval_target_description(memory, row, kind, bbox=None):
    """Generate a fresh description for the exact retrieved target memory.

    This NEVER reuses another memory's description and NEVER uses the personal
    profile's manual description as visual evidence. The exact selected memory
    image is the only visual source. For an object, the matched crop is supplied
    when available so Qwen describes the requested object rather than the scene.
    """
    filename=Path(str(memory.get("image",""))).name
    path=UPLOAD_DIR/filename
    if not path.exists():
        return ""

    try:
        image=load_image(path)
    except Exception as e:
        print("Retrieval image load warning:",e)
        return ""

    visual=image
    if kind=="object" and bbox and len(bbox)==4:
        try:
            visual=crop_object(image,bbox,0.12)
        except Exception:
            visual=image

    target_name=str(row["name"] or "the requested item").strip()

    if qwen_model is not None and qwen_processor is not None:
        if kind=="object":
            prompt=f"""
Describe ONLY the registered personal object "{target_name}" as it appears in this
exact retrieved memory image/crop.

Important:
- Use the current image only. Do not use any previous memory description.
- Do not copy or infer from the user's manual note.
- Describe what is visibly present: object type if visually supported, color, shape,
  visible distinguishing features, and where it is in the current scene.
- Use sensible physical context. Laboratory/electronic items normally sit on a visible
  lab bench/table or are attached to visible equipment; do not call something a sink,
  bathroom surface, floor, wall, shelf, etc. unless that surface is actually visible.
- Do not invent exact model numbers, specifications, labels, functions, or locations.
- If identity is uncertain, say "appears to be" instead of fabricating certainty.
- Do not mention previous images, other memories, or that you are an AI.
Return one concise, natural paragraph.
""".strip()
        else:
            prompt=f"""
Describe ONLY the registered place "{target_name}" as it appears in this exact
retrieved memory image.

Use the current image only. Explain the visible environment, distinctive layout,
furniture/equipment, surfaces and spatial context. Do not copy any previous memory
description or the user's manual note. Do not invent a room or object that is not
visible. Return one concise, natural paragraph.
""".strip()

        try:
            messages=[{"role":"user","content":[
                {"type":"image","image":visual},
                {"type":"text","text":prompt}
            ]}]
            text=qwen_processor.apply_chat_template(
                messages,tokenize=False,add_generation_prompt=True
            )
            try:
                inputs=qwen_processor(
                    text=[text],images=[visual],padding=True,return_tensors="pt"
                )
            except Exception:
                inputs=qwen_processor.apply_chat_template(
                    messages,tokenize=True,add_generation_prompt=True,
                    return_dict=True,return_tensors="pt"
                )
            try:
                inputs=inputs.to(next(qwen_model.parameters()).device)
            except Exception:
                pass
            with torch.no_grad():
                output_ids=qwen_model.generate(
                    **inputs,max_new_tokens=180,do_sample=False,
                    repetition_penalty=1.05,no_repeat_ngram_size=3
                )
            input_ids=inputs.get("input_ids")
            if input_ids is not None:
                generated=[out[len(inp):] for inp,out in zip(input_ids,output_ids)]
                text_out=qwen_processor.batch_decode(
                    generated,skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
            else:
                text_out=qwen_processor.batch_decode(
                    output_ids,skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
            text_out=re.sub(r'\s+',' ',str(text_out or '')).strip()
            if text_out:
                return text_out
        except Exception as e:
            print("Retrieval Qwen description warning:",e)

    # Safe fallback: use only structured evidence already attached to THIS memory.
    if kind=="object":
        for p in memory.get("personal_objects",[]) or []:
            if str(p.get("id",""))==str(row["id"]):
                loc=str(p.get("evidence_source","")).strip()
                return f'{target_name} was visually matched in this saved memory.'
        for p in memory.get("personal_objects",[]) or []:
            if norm_text(p.get("personal_name",""))==norm_text(target_name):
                return f'{target_name} was visually matched in this saved memory.'
    else:
        if memory.get("place"):
            return f'{target_name} was visually matched in this saved memory.'
    return ""


def answer_query(query, memory, results=None):
    if not memory:
        return '🔎 I could not find a confirmed saved sighting for that item.'
    match=memory.get('retrieval_match',{}) or {}
    item=match.get('target_name') or 'the requested item'
    profile_saved_label=match.get('profile_saved_label') or ''
    saved_location=(match.get('target_location') or '').strip()
    saved_room=(match.get('target_room') or '').strip()

    if match.get('target_type')=='personal_profile':
        count=int(memory.get('matched_memory_count') or 0)
        parts=[f'🎯 {item} is the exact object saved in My Objects.']
        if profile_saved_label: parts.append(f'🗓️ Saved: {profile_saved_label}.')
        if saved_location: parts.append(f'📍 Saved location: {saved_location}.')
        if memory.get('profile_description'): parts.append(f'📝 Your description: {memory.get("profile_description")}')
        parts.append(f'📸 {count} confirmed historical picture{"s" if count != 1 else ""} of this same object found.' if count else '📸 No confirmed historical picture of this object was found yet.')
        return ' '.join(parts)

    if match.get('target_type')=='personal_profile':
        parts=[f'🎯 {item} is registered in My Objects/My Places.']
        if saved_location: parts.append(f'📍 Registered location: {saved_location}.')
        if saved_room: parts.append(f'🏠 Room: {saved_room}.')
        if profile_saved_label: parts.append(f'🗓️ Saved on: {profile_saved_label}.')
        parts.append('🖼️ The displayed image is the exact registered reference image.')
        return ' '.join(parts)

    display_location=saved_location or saved_room or (memory.get('area') or '')
    requested_time=memory_time_label(memory)

    # IMPORTANT: The first date/time block is ALWAYS the requested My Object/My Place
    # record. The matched-photo timestamp is shown separately below it. This prevents
    # a historical memory timestamp from being mistaken for the object's own saved
    # date/time.
    profile_date = match.get('profile_saved_date') or ''
    profile_day = match.get('profile_saved_day') or ''
    profile_time = match.get('profile_saved_time') or ''
    parts=[f'🎯 Requested item: {item}.']
    if profile_day and profile_date and profile_time:
        parts.append(f'🗓️ Requested item saved/registered: {profile_day}, {profile_date} at {profile_time}.')
    elif profile_saved_label:
        parts.append(f'🗓️ Requested item saved/registered: {profile_saved_label}.')
    if saved_location:
        parts.append(f'📍 Registered location: {saved_location}.')
    if saved_room:
        parts.append(f'🏠 Registered room: {saved_room}.')
    if requested_time:
        parts.append(f'📸 Confirmed matching picture: {requested_time}.')
    if display_location:
        parts.append(f'📍 Picture location: {display_location}.')
    nearby=[]
    for n in memory.get('nearby_objects',[]) or []:
        name=str(n.get('object','')).strip()
        if name and name.lower() not in {x.lower() for x in nearby}: nearby.append(name.title())
    if nearby: parts.append('👀 Nearby verified: '+', '.join(nearby[:4])+'.')
    target_desc=str(memory.get('retrieval_target_description') or '').strip()
    if target_desc:
        parts.append(f'🤖 AI description: {target_desc}')
    user_note = str(memory.get('retrieval_saved_note') or memory.get('user_description') or '').strip()
    if user_note:
        parts.append(f'📝 Your saved description: {user_note}')
    if results and len(results) > 1:
        earlier_count=len(results)-1
        parts.append(f'🧭 I also found {earlier_count} earlier confirmed sighting{"s" if earlier_count != 1 else ""} of this same item. The history below is only for this requested item.')
    return ' '.join(parts)

# ============================================================
# CHROMA MIGRATION — never deletes old data
# ============================================================

def migrate_chroma_once():
    if MIGRATION_FLAG.exists() or chromadb is None:
        return
    try:
        client = chromadb.PersistentClient(path=str(DATA_DIR))
        old_mem = client.get_or_create_collection(name="echomind_memories")
        old_obj = client.get_or_create_collection(name="personal_objects")
        old_place = client.get_or_create_collection(name="personal_places")
        imported = 0
        with DB_LOCK:
            conn = db()
            existing = {r[0] for r in conn.execute("SELECT id FROM memories").fetchall()}
            data = old_mem.get(include=["metadatas"])
            for md in data.get("metadatas", []):
                mid = md.get("id") or ("memory_" + uuid.uuid4().hex)
                if mid in existing: continue
                image = md.get("image", "")
                if image and (UPLOAD_DIR / Path(image).name).exists():
                    ts = md.get("timestamp", "") or now_local().strftime("%Y-%m-%d %H:%M:%S")
                    try: dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
                    except Exception: dt = now_local()
                    desc = md.get("description", "")
                    obj = json_load(md.get("objects", "[]"), [])
                    pers = json_load(md.get("personal_objects", "[]"), [])
                    place = json_load(md.get("place", "{}"), {})
                    rel = json_load(md.get("relationships", "[]"), [])
                    tags = json_load(md.get("tags", "[]"), [])
                    text = create_memory_text(ts, md.get("area", ""), obj, pers, place, rel, desc, tags)
                    emb = text_embedding(text)
                    conn.execute("""INSERT OR IGNORE INTO memories(id,image,image_url,original_filename,capture_timestamp,date,day,time,original_exif_time,latitude,longitude,gps_accuracy,location_source,area,semantic_place,scene,description,objects_json,personal_objects_json,place_json,relationships_json,tags_json,status,search_text,embedding_json,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (mid, Path(image).name, "/image/"+Path(image).name, "migrated", ts, dt.strftime("%Y-%m-%d"), dt.strftime("%A"), dt.strftime("%H:%M:%S"), "", safe_float(md.get("latitude")), safe_float(md.get("longitude")), None, "Migrated", md.get("area", ""), place.get("name", "") if place else "", "", desc, json_dump(obj), json_dump(pers), json_dump(place), json_dump(rel), json_dump(tags), "migrated", text, json_dump(emb), dt.isoformat(timespec="seconds")))
                    imported += 1

            # Import personal object/place records if SQLite is empty.
            if conn.execute("SELECT COUNT(*) FROM personal_objects").fetchone()[0] == 0:
                data = old_obj.get(include=["metadatas"])
                for md in data.get("metadatas", []):
                    ref = Path(md.get("reference_image", "")).name
                    if not ref: continue
                    conn.execute("INSERT OR IGNORE INTO personal_objects (id,name,owner,location,category,description,reference_image,clip_embedding_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)", ("personal_"+uuid.uuid4().hex, md.get("name",""), md.get("owner",""), md.get("location",""), md.get("category",""), md.get("description",""), ref, md.get("clip_embedding","[]"), now_local().isoformat(timespec="seconds")))
            if conn.execute("SELECT COUNT(*) FROM personal_places").fetchone()[0] == 0:
                data = old_place.get(include=["metadatas"])
                for md in data.get("metadatas", []):
                    ref = Path(md.get("reference_image", "")).name
                    if not ref: continue
                    conn.execute("INSERT OR IGNORE INTO personal_places (id,name,owner,room,category,description,reference_image,clip_embedding_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)", ("place_"+uuid.uuid4().hex, md.get("name",""), md.get("owner",""), md.get("room",""), md.get("category",""), md.get("description",""), ref, md.get("clip_embedding","[]"), now_local().isoformat(timespec="seconds")))
            conn.commit(); conn.close()
        MIGRATION_FLAG.write_text(f"Imported {imported} memories. Existing ChromaDB was not deleted.", encoding="utf-8")
        print(f"Chroma migration complete: {imported} memories imported. Old data was NOT deleted.")
    except Exception as e:
        print("Chroma migration warning:", e)


migrate_chroma_once()
# Backfill BLOBs again after any Chroma import.
sync_reference_blobs()


def repair_reference_profiles():
    """Best-effort startup repair. Existing profile rows/images are never deleted."""
    with DB_LOCK:
        conn=db()
        try:
            for table in ('personal_objects','personal_places'):
                rows=conn.execute(f'SELECT * FROM {table}').fetchall()
                for row in rows:
                    ref_path=REFERENCE_DIR/Path(row['reference_image']).name
                    # Recover the BLOB from disk when possible.
                    try:
                        if not row['reference_image_blob'] and ref_path.exists():
                            conn.execute(f'UPDATE {table} SET reference_image_blob=? WHERE id=?',(sqlite3.Binary(ref_path.read_bytes()),row['id']))
                    except Exception as e:
                        print(f'{table} image repair warning:',e)
                    # Keep the stored embedding if present; missing embeddings are rebuilt lazily.
            conn.commit()
        finally:
            conn.close()


repair_reference_profiles()


def repair_memory_timestamps_from_images():
    """Repair only incomplete legacy timestamps; never overwrite valid saved time or EXIF."""
    with DB_LOCK:
        conn=db()
        try:
            rows=conn.execute("SELECT id,capture_timestamp,created_at,date,day,time,capture_time_source FROM memories").fetchall()
            changed=0
            for r in rows:
                if str(r['date'] or '').strip() and str(r['day'] or '').strip() and str(r['time'] or '').strip():
                    continue
                raw=str(r['capture_timestamp'] or r['created_at'] or '').strip()
                dt=parse_client_datetime(raw)
                capture_value=r['capture_timestamp'] or dt.strftime('%Y-%m-%d %H:%M:%S')
                date_value=dt.strftime('%Y-%m-%d')
                day_value=dt.strftime('%A')
                time_value=dt.strftime('%H:%M:%S')
                source=r['capture_time_source'] or 'Browser/device save time'
                conn.execute("UPDATE memories SET capture_timestamp=?,capture_time_source=?,date=?,day=?,time=? WHERE id=?",(capture_value,source,date_value,day_value,time_value,r['id']))
                changed+=1
            conn.commit()
            print(f"Memory timestamp repair: {changed} incomplete record(s) repaired.")
        finally:
            conn.close()


repair_memory_timestamps_from_images()


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(filename, path, original_filename, browser_time="", latitude=None, longitude=None, gps_accuracy=None, location_source="Unavailable", exif=None, user_description="", is_camera=False):
    """Process a saved image defensively.
    The original file is already on disk before this function runs. Every AI
    subsystem is optional, so a model failure cannot prevent the memory from saving."""
    exif = exif or get_exif_metadata(path)

    if latitude is None:
        latitude = exif.get("latitude")
    if longitude is None:
        longitude = exif.get("longitude")
    if latitude is not None and longitude is not None and location_source == "Unavailable":
        location_source = "EXIF" if exif.get("latitude") is not None else "Browser GPS"

    try:
        area = get_area_name(latitude, longitude)
    except Exception:
        area = f"GPS {latitude:.6f}, {longitude:.6f}" if latitude is not None and longitude is not None else "Location unavailable"

    # Date / Day / Time are the exact browser/device time when the image was saved.
    # Embedded EXIF is preserved separately in original_exif_time for reference only.
    timestamp_dt = parse_client_datetime(browser_time)
    timestamp = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
    capture_time_source = "Browser/device save time"
    user_description = (user_description or "").strip()

    # Each stage is isolated. The uploaded image will still be stored when any
    # detector/model is unavailable or throws an inference error.
    try:
        processed = enhance_for_detection(path)
    except Exception as e:
        print("Preprocessing warning:", e)
        processed = Path(path)

    try:
        detections = detect_objects_precisely(processed)
    except Exception as e:
        traceback.print_exc()
        detections = []

    try:
        ai_description = generate_detailed_description(processed, detections, user_description)
    except Exception as e:
        traceback.print_exc()
        ai_description = "Automatic visual description unavailable for this image."

    description = ai_description

    try:
        relationships = build_spatial_context(detections)
    except Exception as e:
        print("Spatial reasoning warning:", e)
        relationships = []

    try:
        image = load_image(path)
    except Exception as e:
        raise RuntimeError(f"Saved image could not be reopened: {e}") from e

    try:
        personal_matches = match_personal_objects(image, detections)
    except Exception as e:
        traceback.print_exc()
        personal_matches = []

    try:
        nearby_objects = nearby_objects_for_personal_matches(personal_matches, detections, 3, image.size)
        target_location = ''
        if personal_matches:
            target_location = str(personal_matches[0].get('location') or '').strip()
        nearby_objects = filter_nearby_context_objects(
            nearby_objects, area, '', target_location
        )[:3]
    except Exception as e:
        print("Nearby-object warning:", e)
        nearby_objects = []


    try:
        place = match_personal_place(image)
    except Exception as e:
        traceback.print_exc()
        place = None

    try:
        tags = generate_tags(detections, personal_matches, place, relationships, area)
    except Exception:
        tags = []

    # Cache ONLY verified detector regions. Arbitrary grid/full-image CLIP regions are
    # intentionally not stored because they can make unrelated memories look similar.
    clip_regions = []
    if clip_model is not None and detections:
        try:
            candidates = []
            for det in detections:
                try:
                    candidates.append((crop_object(image, det['bbox'], 0.08), det))
                except Exception:
                    continue
            if candidates:
                embs = clip_image_batch([x[0] for x in candidates])
                for (crop, det), emb in zip(candidates, embs):
                    if emb:
                        clip_regions.append({
                            "embedding": emb,
                            "bbox": det.get("bbox"),
                            "source": "YOLO-region",
                            "object_name": det.get("name", ""),
                        })
        except Exception as e:
            print("Memory visual cache warning:", e)

    # Persistence is the final non-optional step.
    saved=save_memory(
        filename, original_filename, timestamp, exif.get("taken_at", ""),
        latitude, longitude, gps_accuracy, location_source, area,
        detections, personal_matches, place, relationships,
        description, tags, user_description, clip_regions, nearby_objects, capture_time_source
    )
    saved['user_friendly_evidence']=build_user_friendly_evidence(saved)
    return saved


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template_string(PAGE, page="home", memory=None, error=None, objects=list_personal_objects(), places=list_personal_places())


@app.route("/upload", methods=["POST"])
def upload():
    try:
        filename, path = save_uploaded_file(request.files.get("image"))
        exif = get_exif_metadata(path)
        lat = safe_float(request.form.get("latitude"))
        lon = safe_float(request.form.get("longitude"))
        accuracy = safe_float(request.form.get("gps_accuracy"))
        if lat is None: lat = exif.get("latitude")
        if lon is None: lon = exif.get("longitude")
        source = "Browser GPS" if request.form.get("latitude") else ("EXIF" if exif.get("latitude") is not None else "Unavailable")
        memory = process_image(filename, path, request.files.get("image").filename, request.form.get("browser_time", ""), lat, lon, accuracy, source, exif, request.form.get("user_description", ""), is_camera=False)
        return render_template_string(PAGE, page="home", memory=memory, error=None, objects=list_personal_objects(), places=list_personal_places())
    except Exception as e:
        print("UPLOAD ERROR:", e)
        return render_template_string(PAGE, page="home", memory=None, error=f"Upload failed: {e}", objects=list_personal_objects(), places=list_personal_places()), 400


@app.route("/camera", methods=["POST"])
def camera():
    try:
        filename, path = save_base64_image(request.form.get("image_data"))
        lat = safe_float(request.form.get("latitude")); lon = safe_float(request.form.get("longitude")); acc = safe_float(request.form.get("gps_accuracy"))
        source = "Browser GPS" if lat is not None and lon is not None else "Unavailable"
        memory = process_image(filename, path, "camera_capture.jpg", request.form.get("browser_time", ""), lat, lon, acc, source, get_exif_metadata(path), is_camera=True)
        return render_template_string(PAGE, page="home", memory=memory, error=None, objects=list_personal_objects(), places=list_personal_places())
    except Exception as e:
        print("CAMERA ERROR:", e)
        return render_template_string(PAGE, page="home", memory=None, error=f"Camera capture failed: {e}", objects=list_personal_objects(), places=list_personal_places()), 400


@app.route("/objects")
def objects():
    try:
        return render_template_string(PAGE,page="objects",memory=None,error=None,objects=list_personal_objects(),places=list_personal_places(),memories=[])
    except Exception as e:
        traceback.print_exc()
        return f"EchoMind Objects page error: {type(e).__name__}: {e}",500


@app.route("/objects/add", methods=["POST"])
def objects_add():
    try:
        category=request.form.get("category","").strip()
        if category=="__OTHER__": category=request.form.get("category_custom","").strip()
        save_personal_object(request.form.get("name",""),request.form.get("owner",""),request.form.get("location",""),category,request.form.get("description",""),request.files.get("reference_image"),request.form.get("browser_time",""))
        return redirect(url_for("objects"))
    except Exception as e:
        traceback.print_exc()
        return render_template_string(PAGE,page="objects",memory=None,error=f"Could not save personal object: {type(e).__name__}: {e}",objects=list_personal_objects(),places=list_personal_places(),memories=[]),200


@app.route("/places")
def places():
    try:
        return render_template_string(PAGE,page="places",memory=None,error=None,objects=list_personal_objects(),places=list_personal_places(),memories=[])
    except Exception as e:
        traceback.print_exc()
        return f"EchoMind Places page error: {type(e).__name__}: {e}",500


@app.route("/places/add", methods=["POST"])
def places_add():
    try:
        category=request.form.get("category","").strip()
        if category=="__OTHER__": category=request.form.get("category_custom","").strip()
        save_personal_place(request.form.get("name",""),request.form.get("owner",""),request.form.get("room",""),category,request.form.get("description",""),request.files.get("reference_image"),request.form.get("browser_time",""))
        return redirect(url_for("places"))
    except Exception as e:
        traceback.print_exc()
        return render_template_string(PAGE,page="places",memory=None,error=f"Could not save personal place: {type(e).__name__}: {e}",objects=list_personal_objects(),places=list_personal_places(),memories=[]),200


@app.route("/retrieve", methods=["GET", "POST"])
def retrieve():
    query=request.form.get("query","").strip() if request.method=="POST" else ""
    try:
        results=retrieve_memories(query,5) if query else []
        result=results[0] if results else None
        answer=answer_query(query,result,results) if result else ("" if not query else "I could not find a relevant saved memory.")
        return render_template_string(PAGE,page="retrieve",memory=None,result=result,results=results,answer=answer,query=query,error=None,objects=list_personal_objects(),places=list_personal_places(),memories=[])
    except Exception as e:
        traceback.print_exc()
        return render_template_string(PAGE,page="retrieve",memory=None,result=None,results=[],answer="EchoMind could not complete this search. Your saved data was not deleted.",query=query,error=f"Retrieval error: {type(e).__name__}: {e}",objects=list_personal_objects(),places=list_personal_places(),memories=[]),200


@app.route("/objects/delete/<object_id>", methods=["POST"])
def delete_personal_object(object_id):
    with DB_LOCK:
        conn=db()
        row=conn.execute("SELECT reference_image FROM personal_objects WHERE id=?",(object_id,)).fetchone()
        if not row:
            conn.close()
            return redirect(url_for("objects"))

        # Remove the profile first so no NEW image can ever match it again.
        conn.execute("DELETE FROM personal_objects WHERE id=?",(object_id,))

        # Also remove only this deleted profile's historical personal-match metadata
        # from saved memories. The original memory image and generic YOLO detections
        # stay untouched. This prevents a deleted profile from reappearing everywhere.
        rows=conn.execute("SELECT id,personal_objects_json,tags_json,search_text FROM memories").fetchall()
        for m in rows:
            personal=json_load(m["personal_objects_json"],[])
            filtered=[x for x in personal if str(x.get("id", "")) != str(object_id)]
            if len(filtered) == len(personal):
                continue
            tags=json_load(m["tags_json"],[])
            deleted_names={str(x.get("personal_name", "")) for x in personal if str(x.get("id", "")) == str(object_id)}
            tags=[t for t in tags if str(t) not in deleted_names]
            # Rebuild the searchable text from remaining structured memory data.
            mem_row=conn.execute("SELECT * FROM memories WHERE id=?",(m["id"],)).fetchone()
            if mem_row:
                mem=row_to_memory(mem_row)
                search=create_memory_text(
                    mem.get("capture_timestamp", ""), mem.get("area", ""), mem.get("objects", []),
                    filtered, mem.get("place", {}), mem.get("relationships", []),
                    mem.get("description", ""), tags, mem.get("nearby_objects", [])
                )
                conn.execute(
                    "UPDATE memories SET personal_objects_json=?,tags_json=?,search_text=? WHERE id=?",
                    (json_dump(filtered),json_dump(tags),search,m["id"])
                )
        conn.commit()
        conn.close()
    try:
        p=REFERENCE_DIR/Path(row["reference_image"]).name
        if p.exists(): p.unlink()
    except Exception: pass
    return redirect(url_for("objects"))

@app.route("/places/delete/<place_id>", methods=["POST"])
def delete_personal_place(place_id):
    with DB_LOCK:
        conn=db(); row=conn.execute("SELECT reference_image FROM personal_places WHERE id=?",(place_id,)).fetchone()
        if not row: conn.close(); return redirect(url_for("places"))
        conn.execute("DELETE FROM personal_places WHERE id=?",(place_id,)); conn.commit(); conn.close()
    try:
        p=REFERENCE_DIR/Path(row["reference_image"]).name
        if p.exists(): p.unlink()
    except Exception: pass
    return redirect(url_for("places"))

@app.route("/database")
def database():
    return render_template_string(PAGE, page="database", memory=None, memories=all_memories(), error=None, objects=list_personal_objects(), places=list_personal_places())


@app.route("/api/memories")
def api_memories():
    return jsonify({"memories": all_memories()})


@app.route("/memory/delete/<memory_id>", methods=["POST"])
def delete_memory(memory_id):
    with DB_LOCK:
        conn = db(); row = conn.execute("SELECT image FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            conn.close(); return jsonify({"ok": False, "error": "Memory not found"}), 404
        conn.execute("DELETE FROM memories WHERE id=?", (memory_id,)); conn.commit(); conn.close()
    # Delete only the stored uploaded image and its generated processed copy.
    filename = Path(row["image"]).name
    for p in (UPLOAD_DIR / filename, UPLOAD_DIR / (Path(filename).stem + "_processed.jpg")):
        try:
            if p.exists(): p.unlink()
        except Exception as e: print("Delete image warning:", e)
    return redirect(url_for("database"))


@app.route("/memory-image/<memory_id>")
def memory_image(memory_id):
    """Serve the exact image linked to one SQLite memory record."""
    safe_id=re.sub(r"[^a-zA-Z0-9_-]", "", str(memory_id or ""))
    if not safe_id:
        return "Memory image not found",404
    with DB_LOCK:
        conn=db()
        row=conn.execute("SELECT image FROM memories WHERE id=?", (safe_id,)).fetchone()
        conn.close()
    if not row:
        return "Memory image not found",404
    filename=Path(str(row["image"])).name
    path=UPLOAD_DIR/filename
    if not path.exists():
        return "Stored memory image not found",404
    response=send_from_directory(str(UPLOAD_DIR), filename, as_attachment=False, conditional=False)
    response.headers["Cache-Control"]="no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"]="no-cache"
    response.headers["Expires"]="0"
    return response


@app.route("/image/<path:filename>")
def image_file(filename):
    """Serve the exact stored upload image directly from the uploads directory."""
    safe_name = Path(str(filename or "")).name
    if not safe_name:
        return "Image not found", 404
    path = UPLOAD_DIR / safe_name
    if not path.exists() or not path.is_file():
        return "Stored image not found", 404
    ext = path.suffix.lower()
    mime = {
        ".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".png":"image/png",
        ".webp":"image/webp", ".gif":"image/gif", ".bmp":"image/bmp",
        ".tif":"image/tiff", ".tiff":"image/tiff"
    }.get(ext, "application/octet-stream")
    response = send_file(path, mimetype=mime, as_attachment=False, conditional=False, max_age=0)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/stored-memory-image/<memory_id>")
def stored_memory_image(memory_id):
    """Serve a memory image by its authoritative SQLite memory ID."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(memory_id or ""))
    if not safe_id:
        return "Memory image not found", 404
    with DB_LOCK:
        conn = db()
        try:
            row = conn.execute("SELECT image FROM memories WHERE id=?", (safe_id,)).fetchone()
        finally:
            conn.close()
    if not row or not row["image"]:
        return "Memory image not found", 404
    filename = Path(str(row["image"])).name
    path = UPLOAD_DIR / filename
    if not path.exists() or not path.is_file():
        return "Stored memory image not found", 404
    ext = path.suffix.lower()
    mime = {
        ".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".png":"image/png",
        ".webp":"image/webp", ".gif":"image/gif", ".bmp":"image/bmp",
        ".tif":"image/tiff", ".tiff":"image/tiff"
    }.get(ext, "application/octet-stream")
    response = send_file(path, mimetype=mime, as_attachment=False, conditional=False, max_age=0)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/reference/<path:filename>")
def reference_file(filename):
    return send_from_directory(str(REFERENCE_DIR), Path(filename).name, as_attachment=False)


@app.route("/personal-reference/<kind>/<profile_id>")
def personal_reference(kind, profile_id):
    table = "personal_objects" if kind == "object" else "personal_places" if kind == "place" else None
    if table is None:
        return "Not found", 404
    with DB_LOCK:
        conn = db()
        try:
            row = conn.execute(
                f"SELECT reference_image, reference_image_blob FROM {table} WHERE id=?",
                (profile_id,)
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return "Not found", 404
    ext = Path(row["reference_image"]).suffix.lower()
    mime = {
        ".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".png":"image/png",
        ".webp":"image/webp", ".gif":"image/gif", ".bmp":"image/bmp",
        ".tif":"image/tiff", ".tiff":"image/tiff"
    }.get(ext, "image/jpeg")
    if row["reference_image_blob"]:
        response = make_response(bytes(row["reference_image_blob"]))
    else:
        path = REFERENCE_DIR / Path(row["reference_image"]).name
        if not path.exists():
            return "Reference image not found", 404
        response = make_response(path.read_bytes())
    response.headers["Content-Type"] = mime
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.errorhandler(500)
def internal_server_error(error):
    # Keep the 500 handler independent from PAGE so a template failure cannot
    # trigger another template failure and hide the real exception.
    traceback.print_exc()
    message = f"EchoMind Internal Server Error: {type(error).__name__}: {error}"
    html = '<html><body style="font-family:Arial;padding:30px;background:#eef2ff">' + f'<h1>EchoMind Internal Server Error</h1><p>{message}</p><p>See the terminal traceback for the exact failing line.</p>' + '</body></html>'
    return html, 500


def profile_time_label(profile):
    """Format a personal-profile saved timestamp for both dicts and sqlite3.Row."""
    try:
        if hasattr(profile, 'keys') and 'created_at' in profile.keys():
            raw = str(profile['created_at'] or '').strip()
        elif isinstance(profile, dict):
            raw = str(profile.get('created_at') or '').strip()
        else:
            raw = ''
    except Exception:
        raw = ''
    if not raw:
        return 'Registration date/time unavailable'
    try:
        return parse_client_datetime(raw).strftime('%A, %Y-%m-%d at %H:%M:%S')
    except Exception:
        try:
            return datetime.fromisoformat(raw).strftime('%A, %Y-%m-%d at %H:%M:%S')
        except Exception:
            return 'Registration date/time unavailable'

app.jinja_env.globals['profile_time_label']=profile_time_label

# ============================================================
# UI
# ============================================================

PAGE = r'''
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EchoMind — Memory Assistant</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="anonymous">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin="anonymous"></script>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Inter,Arial,sans-serif;background:#eef2ff;color:#172033}.wrap{max-width:1250px;margin:auto;padding:22px}.hero{background:linear-gradient(135deg,#172554,#2563eb);color:#fff;border-radius:24px;padding:28px;text-align:center}.hero h1{font-size:42px;margin:0 0 6px}.nav{display:flex;flex-wrap:wrap;gap:9px;justify-content:center;margin-top:20px}.nav a{color:#fff;text-decoration:none;padding:10px 14px;background:rgba(255,255,255,.15);border-radius:10px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}.card{background:#fff;margin-top:20px;padding:24px;border-radius:18px;box-shadow:0 8px 30px rgba(15,23,42,.08)}label{font-weight:700;display:block;margin-top:8px}input,textarea{width:100%;padding:12px;margin:7px 0 15px;border:1px solid #cbd5e1;border-radius:10px}button{border:0;border-radius:10px;padding:12px 17px;background:#2563eb;color:#fff;cursor:pointer;font-weight:700}button.secondary{background:#475569}button.danger{background:#dc2626}.small{color:#64748b;font-size:13px}.ok{padding:16px;background:#ecfdf5;border-radius:12px;color:#065f46}.error{padding:16px;background:#fef2f2;border-radius:12px;color:#991b1b}.answer{padding:18px;background:#ecfdf5;border-radius:14px;font-size:18px;line-height:1.65}.preview{width:100%;max-height:560px;object-fit:contain;border-radius:14px;background:#f8fafc;margin-top:12px}.tag{display:inline-block;background:#dbeafe;border-radius:20px;padding:6px 10px;margin:3px;font-size:13px}.object{background:#f8fafc;border-radius:12px;padding:14px;margin:10px 0}.memory-img{width:180px;height:140px;object-fit:cover;border-radius:12px}.two{display:grid;grid-template-columns:190px 1fr;gap:18px;align-items:start}video{width:100%;max-height:500px;background:#111;border-radius:14px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #e2e8f0;vertical-align:top}.pill{font-size:12px;padding:4px 8px;border-radius:20px;background:#e2e8f0}.status{margin-top:10px;font-weight:700}.evidence-box{background:#f8fafc;border-left:4px solid #2563eb;font-size:15px}.time-grid{grid-template-columns:repeat(3,minmax(150px,1fr))}.time-card{text-align:center;padding:18px;border:1px solid #dbeafe;background:linear-gradient(180deg,#ffffff,#eff6ff);box-shadow:0 6px 18px rgba(37,99,235,.08)}.time-icon{font-size:25px;margin-bottom:5px}.time-card strong{font-size:18px;display:block;margin-top:5px}.matched-memory-card{position:relative;border:3px solid #7c3aed;background:linear-gradient(180deg,#faf5ff,#fff);box-shadow:0 12px 35px rgba(124,58,237,.18);padding:18px;border-radius:18px}.matched-memory-badge{display:inline-flex;align-items:center;gap:7px;background:#7c3aed;color:#fff;border-radius:999px;padding:7px 12px;font-size:13px;font-weight:800;margin-bottom:10px}.matched-memory-image{width:100%;max-height:620px;object-fit:contain;border:4px solid #a855f7;border-radius:16px;background:#f8fafc}.matched-time-panel{border:2px solid #7c3aed;background:linear-gradient(135deg,#faf5ff,#eff6ff);box-shadow:0 10px 28px rgba(124,58,237,.12);padding:20px;border-radius:18px}.matched-time-title{font-size:20px;font-weight:850;color:#4c1d95;margin-bottom:5px}.matched-time-subtitle{color:#64748b;font-size:13px;margin-bottom:15px}.location-card{font-size:18px;background:#f8fafc;border-left:4px solid #10b981}.history-row{padding:5px 0}
@media(max-width:700px){.time-grid{grid-template-columns:1fr!important}.matched-memory-card,.matched-time-panel{padding:14px}.matched-memory-image{max-height:420px}}\n
.reference-card{margin-top:18px;padding:18px;border:2px solid #bfdbfe;border-radius:18px;background:linear-gradient(180deg,#f8fbff,#fff);box-shadow:0 8px 24px rgba(30,64,175,.08)}.reference-header{display:flex;justify-content:space-between;gap:12px;align-items:center;color:#1e3a8a;font-size:15px}.reference-header span{font-size:11px;font-weight:900;letter-spacing:.8px;padding:6px 9px;border-radius:999px;background:#dbeafe;color:#1d4ed8}.reference-body{display:grid;grid-template-columns:minmax(150px,240px) 1fr;gap:18px;align-items:center;margin-top:13px}.reference-image{width:100%;height:210px;object-fit:contain;background:#fff;border:2px solid #dbeafe;border-radius:14px}.reference-copy{min-width:0}.reference-title{font-size:14px;color:#64748b;line-height:1.5}.reference-name{font-size:23px;font-weight:900;color:#1d4ed8;margin-top:5px}.reference-description{margin-top:8px;padding:11px 13px;background:#eff6ff;border-radius:12px;color:#334155;line-height:1.55}.reference-note{margin-top:9px;font-size:13px;color:#64748b;line-height:1.5}.map-caption{font-size:13px;color:#64748b;margin:-2px 0 10px}.map-shell{padding:0!important;overflow:hidden;border:1px solid #cbd5e1;box-shadow:0 12px 30px rgba(15,23,42,.08)}#memoryPathMap{height:430px;width:100%;background:#dfe7ef}.leaflet-control-zoom{border:1px solid #cbd5e1!important;box-shadow:0 4px 14px rgba(15,23,42,.12)!important}.leaflet-control-attribution{font-size:10px!important}@media(max-width:700px){.reference-body{grid-template-columns:1fr}.reference-image{height:230px}.reference-header{align-items:flex-start;flex-direction:column}.map-shell #memoryPathMap{height:340px}}

.retrieval-shell{overflow:hidden}.section-kicker{display:inline-flex;align-items:center;padding:7px 12px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:12px;font-weight:900;letter-spacing:.5px;margin-bottom:8px}.profile-primary-card{border:3px solid #f59e0b;background:linear-gradient(145deg,#fff7ed,#ffffff 55%,#f5f3ff);box-shadow:0 18px 50px rgba(245,158,11,.15)}.profile-image-wrap{border-color:#f59e0b;background:#fffbeb}.primary-profile-status{margin-top:22px;padding:15px 18px;border-radius:16px;background:linear-gradient(135deg,#ecfdf5,#eff6ff);border:1px solid #86efac;color:#166534;font-size:14px}.single-history-note{margin-top:18px;padding:16px 18px;border-radius:16px;background:linear-gradient(135deg,#f8fafc,#eef2ff);border:1px solid #cbd5e1;color:#334155;font-weight:700}.primary-result-card{margin-top:20px;padding:24px;border:3px solid #2563eb;border-radius:24px;background:linear-gradient(145deg,#eff6ff,#fff);box-shadow:0 14px 40px rgba(37,99,235,.13)}.primary-title,.history-title{margin:0 0 4px;color:#0f172a}.target-name{font-size:28px;font-weight:900;color:#1d4ed8;margin:6px 0 8px}.section-subtitle{color:#64748b;font-size:14px;line-height:1.55;margin:6px 0 18px}.primary-image-wrap{position:relative;border:4px solid #7c3aed;border-radius:20px;padding:8px;background:#faf5ff;box-shadow:0 14px 35px rgba(124,58,237,.17);margin-bottom:22px}.image-badge{position:absolute;left:16px;top:16px;z-index:2;background:#7c3aed;color:#fff;border-radius:999px;padding:8px 13px;font-weight:900;font-size:12px;box-shadow:0 6px 16px rgba(0,0,0,.12)}.primary-memory-image{display:block;width:100%;max-height:650px;object-fit:contain;border-radius:14px;background:#f8fafc}.detail-section{margin-top:20px}.detail-heading{font-size:19px;font-weight:900;margin-bottom:9px;color:#172033}.detail-card{border-radius:16px;padding:16px 18px;line-height:1.65;font-size:16px}.manual-card{background:#fff;border:2px solid #93c5fd;box-shadow:0 6px 18px rgba(37,99,235,.07)}.ai-card{background:#fff;border:2px solid #c4b5fd;box-shadow:0 6px 18px rgba(124,58,237,.07)}.datetime-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.datetime-box{padding:18px;border-radius:18px;text-align:center;background:#fff;border:2px solid #60a5fa;box-shadow:0 8px 22px rgba(37,99,235,.10)}.datetime-icon{font-size:29px}.datetime-label{font-size:12px;font-weight:900;color:#64748b;letter-spacing:.8px;margin-top:5px}.datetime-value{font-size:20px;font-weight:900;color:#1e3a8a;margin-top:5px}.context-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.context-card{background:#f8fafc;border-left:5px solid #10b981;border-radius:12px;padding:15px}.context-label,.history-label{font-size:12px;font-weight:900;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}.context-value{font-size:16px;font-weight:700}.nearby-grid{display:flex;flex-wrap:wrap;gap:9px}.nearby-chip{background:#eef6ff;border:1px solid #bfdbfe;padding:9px 12px;border-radius:999px;font-weight:700}.nearby-chip span{color:#64748b;font-weight:600;font-size:12px}.history-section,.roadmap-section{margin-top:26px;padding:24px;border-radius:24px;background:#fff;border:1px solid #dbeafe;box-shadow:0 8px 30px rgba(15,23,42,.07)}.history-card{display:grid;grid-template-columns:44px 1fr;gap:14px;padding:18px;margin-top:16px;border:2px solid #e2e8f0;border-radius:20px;background:#fbfdff}.history-number{width:38px;height:38px;border-radius:50%;background:#e2e8f0;display:grid;place-items:center;font-weight:900}.history-content{min-width:0}.history-image{width:100%;max-height:440px;object-fit:contain;background:#f8fafc;border-radius:14px;margin-bottom:15px;border:2px solid #e2e8f0}.history-heading{font-size:18px;font-weight:900;margin-bottom:12px;color:#334155}.history-detail{padding:12px 14px;background:#f8fafc;border-radius:12px;margin-top:10px}.history-text{font-size:15px;line-height:1.55}.history-gps{font-size:13px;color:#64748b;margin-top:5px}.mini-datetime-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.mini-datetime-grid>div{padding:10px;background:#fff;border-radius:10px;border:1px solid #dbeafe}.mini-datetime-grid b{display:block;font-size:12px;color:#64748b}.mini-datetime-grid span{display:block;margin-top:3px;font-weight:800;color:#1e3a8a}.roadmap{position:relative;margin-top:18px;padding-left:34px}.roadmap:before{content:"";position:absolute;left:15px;top:10px;bottom:10px;width:4px;background:#dbeafe;border-radius:99px}.roadmap-node{position:relative;padding:0 0 18px}.node-dot{position:absolute;left:-29px;top:5px;width:28px;height:28px;border-radius:50%;background:#7c3aed;color:#fff;display:grid;place-items:center;font-size:14px;font-weight:900;border:3px solid #fff;box-shadow:0 0 0 2px #7c3aed;z-index:1}.secondary-node-dot{background:#64748b;box-shadow:0 0 0 2px #64748b}.node-card{padding:14px 16px;border-radius:14px;background:#f8fafc;border:1px solid #e2e8f0}.primary-node-card{background:#faf5ff;border:2px solid #c4b5fd}.node-title{font-size:13px;font-weight:900;letter-spacing:.4px}.primary-node-card .node-title{color:#6d28d9}.node-meta{margin-top:4px;color:#1e3a8a;font-weight:800}.node-location{margin-top:5px;color:#475569;font-size:14px}@media(max-width:700px){.datetime-grid,.mini-datetime-grid{grid-template-columns:1fr}.primary-result-card,.history-section,.roadmap-section{padding:16px}.target-name{font-size:24px}.history-card{grid-template-columns:32px 1fr}.history-number{width:30px;height:30px;font-size:12px}.node-meta{font-size:14px}}

</style>
</head>
<body><div class="wrap">
<div class="hero"><h1>EchoMind</h1><div>Vision-Language Episodic Memory Retrieval System</div><div class="small" style="color:#dbeafe">Personalized visual memory assistance • Object-level recognition • Time • Location • Retrieval</div><div class="nav">
<a href="/">📷 Capture Memory</a><a href="/objects">🎒 My Objects</a><a href="/places">🏠 My Places</a><a href="/retrieve">🧠 Ask EchoMind</a><a href="/database">🗄️ Memory Database</a>
</div></div>
{% if error %}<div class="card error">{{error}}</div>{% endif %}

{% if page == 'home' %}
<div class="grid">
<div class="card"><h2>📤 Upload a Memory Image</h2><p class="small">The original image is saved first. AI processing happens only after the file is safely stored.</p>
<form method="POST" action="/upload" enctype="multipart/form-data" onsubmit="stampUpload()">
<input type="file" name="image" accept="image/*" required><label>📝 Your Description (Optional)</label><textarea name="user_description" rows="4" placeholder="Add your own details about this memory."></textarea>
<input type="hidden" name="latitude" id="upload_latitude"><input type="hidden" name="longitude" id="upload_longitude"><input type="hidden" name="gps_accuracy" id="upload_accuracy"><input type="hidden" name="browser_time" id="upload_time">
<button type="button" class="secondary" onclick="getUploadLocation()">📍 Capture GPS</button> <button>Analyze & Save Memory</button><div id="uploadStatus" class="status"></div></form></div>
<div class="card"><h2>🎥 Real-Time Camera</h2><p class="small">Works on localhost/HTTPS after browser camera permission.</p><button type="button" class="secondary" onclick="startCamera()">Allow Camera</button> <button type="button" onclick="captureImage()">Capture Current View</button><video id="camera" autoplay playsinline></video><canvas id="canvas" style="display:none"></canvas>
<form method="POST" action="/camera" id="cameraForm"><input type="hidden" name="image_data" id="image_data"><input type="hidden" name="latitude" id="cam_lat"><input type="hidden" name="longitude" id="cam_lon"><input type="hidden" name="gps_accuracy" id="cam_acc"><input type="hidden" name="browser_time" id="cam_time"></form><div id="cameraStatus" class="status"></div></div>
</div>
{% if memory %}<div class="card"><div class="ok"><strong>✓ MEMORY SAVED</strong><br>Image uploaded and stored successfully in the persistent EchoMind database.</div>
<h2>Saved Image</h2><img class="preview" src="{{url_for('stored_memory_image',memory_id=memory.id)}}"><p><a href="{{url_for('stored_memory_image',memory_id=memory.id)}}" target="_blank">Open stored image</a></p>
<h3>🧠 AI-Generated Description</h3><div class="object evidence-box" style="white-space:pre-wrap;line-height:1.6">{{memory.description}}</div>
<h3>Date / Day / Time</h3><p><b>{{memory.date}}</b> • {{memory.day}} • <b>{{memory.time}}</b></p><p>Application timestamp: {{memory.capture_timestamp}}{% if memory.original_exif_time %}<br>Original EXIF time: {{memory.original_exif_time}}{% endif %}</p>
<h3>Location</h3><p>{{memory.area}}<br>Source: {{memory.location_source}}{% if memory.latitude is not none %}<br>GPS: {{'%.6f'|format(memory.latitude)}}, {{'%.6f'|format(memory.longitude)}}{% endif %}</p>
<h3>✨ Evidence</h3><div class="object evidence-box" style="white-space:pre-wrap;line-height:1.55;font-size:16px">{{memory.user_friendly_evidence or "🔎 No additional verified evidence."}}<div style="margin-top:14px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px"><div><div class="small"><b>Exact saved image</b></div><img class="preview" style="max-height:320px" src="{{url_for('stored_memory_image',memory_id=memory.id)}}" alt="Exact saved memory evidence"></div>{% for p in memory.personal_objects[:1] %}<div><div class="small"><b>Registered object reference</b></div><img class="preview" style="max-height:320px" src="{{url_for('personal_reference',kind='object',profile_id=p.id)}}" alt="Registered personal object evidence"></div>{% endfor %}</div></div>
<h3>🔍 Confirmed Visible Objects</h3>{% for o in memory.objects %}<div class="object"><b>{{o.name|title}}</b>{% if o.description %}<br>{{o.description}}{% endif %}</div>{% else %}<p class="small">No additional objects were confidently confirmed.</p>{% endfor %}
{% if memory.personal_objects %}<h3>Personal Object Matches</h3>{% for p in memory.personal_objects %}<div class="object"><b>{{p.personal_name}}</b> • {{p.owner}} • similarity {{p.similarity}}</div>{% endfor %}{% endif %}
{% if memory.nearby_objects %}<h3>Nearby Objects (Verified in This Image)</h3>{% for n in memory.nearby_objects[:3] %}<div class="object"><b>{{n.object|title}}</b> • {{n.position}} • {{n.distance_pixels}} px away</div>{% endfor %}{% endif %}
{% if memory.place %}<h3>Personal Place Match</h3><div class="object"><b>{{memory.place.name}}</b> • {{memory.place.room}} • similarity {{memory.place.similarity}}</div>{% endif %}
<h3>Spatial Relationships</h3>{% for r in memory.relationships[:60] %}<div class="object"><b>{{r.object}}</b> → {{r.relation|join(', ')}} → <b>{{r.reference}}</b></div>{% endfor %}
<h3>Tags</h3>{% for t in memory.tags %}<span class="tag">{{t}}</span>{% endfor %}</div>{% endif %}
{% endif %}

{% if page == 'objects' %}<div class="card"><h2>🎒 My Objects</h2><p class="small">Register personal items such as “Isra's black charger”, keys, glasses or a wallet. The reference image is permanently stored.</p><form method="POST" action="/objects/add" enctype="multipart/form-data" onsubmit="stampProfileTime(this)"><input type="hidden" name="browser_time"><label>📷 Reference Image <span class="small">(upload first)</span></label><input type="file" name="reference_image" accept="image/*" required><label>Name</label><input name="name" required placeholder="Isra's black charger"><label>Owner</label><input name="owner" placeholder="Isra"><label>📍 Location</label><input name="location" placeholder="e.g. Bedroom desk, Faculty Room C-14, Kitchen shelf"><label>Category</label><select name="category" required onchange="toggleCustom(this,'object_category_custom')"><option value="">Select a category…</option><option>Electronics & Gadgets</option><option>Personal Items</option><option>Documents & Stationery</option><option>Clothing & Accessories</option><option>Bags & Luggage</option><option>Food & Lunch</option><option>Kitchen & Dining</option><option>Medicine & Health</option><option>Cleaning & Household</option><option>Office & Work</option><option>Academic & Study</option><option>Keys & Access Cards</option><option value="__OTHER__">Other / Custom</option></select><input id="object_category_custom" name="category_custom" placeholder="Write your own category (for Other)" style="display:none"><label>Description</label><textarea name="description" placeholder="Black charger with white connector"></textarea><button>Save Personal Object</button></form></div><div class="card"><h3>Saved Personal Objects</h3>{% for o in objects %}<div class="two object"><img class="memory-img" src="{{url_for('personal_reference',kind='object',profile_id=o.id)}}"><div><b>{{o.name}}</b><br>Owner: {{o.owner}}<br>Location: {{o.location}}<br>Category: {{o.category}}<br>{{o.description}}<br><span class="pill">🗓️ Saved: {{profile_time_label(o)}}</span><form method="POST" action="{{url_for('delete_personal_object',object_id=o.id)}}" onsubmit="return confirm('Delete this personal object?')"><button class="danger">Delete Object</button></form></div></div>{% else %}<p>No personal objects registered yet.</p>{% endfor %}</div>{% endif %}

{% if page == 'places' %}<div class="card"><h2>🏠 My Places</h2><p class="small">Register important personal places/rooms with reference photos for personalized visual matching.</p><form method="POST" action="/places/add" enctype="multipart/form-data" onsubmit="stampProfileTime(this)"><input type="hidden" name="browser_time"><label>📷 Reference Image <span class="small">(upload first)</span></label><input type="file" name="reference_image" accept="image/*" required><label>Place Name</label><input name="name" required placeholder="Isra's bedroom"><label>Owner</label><input name="owner" placeholder="Isra"><label>Room / Area</label><input name="room" placeholder="Bedroom"><label>Category</label><select name="category" required onchange="toggleCustom(this,'place_category_custom')"><option value="">Select a category…</option><option>Home / Residence</option><option>Office / Workplace</option><option>Classroom / Lecture Room</option><option>Laboratory</option><option>Hospital / Clinic</option><option>Library</option><option>Kitchen / Dining</option><option>Outdoor / Public Space</option><option>Event / Venue</option><option value="__OTHER__">Other / Custom</option></select><input id="place_category_custom" name="category_custom" placeholder="Write your own category (for Other)" style="display:none"><label>Description</label><textarea name="description" placeholder="Study table near window"></textarea><button>Save Personal Place</button></form></div><div class="card"><h3>Saved Personal Places</h3>{% for p in places %}<div class="two object"><img class="memory-img" src="{{url_for('personal_reference',kind='place',profile_id=p.id)}}"><div><b>{{p.name}}</b><br>Owner: {{p.owner}}<br>Room: {{p.room}}<br>{{p.description}}<br><span class="pill">🗓️ Saved: {{profile_time_label(p)}}</span><form method="POST" action="{{url_for('delete_personal_place',place_id=p.id)}}" onsubmit="return confirm('Delete this personal place?')"><button class="danger">Delete Place</button></form></div></div>{% else %}<p>No personal places registered yet.</p>{% endfor %}</div>{% endif %}

{% if page == 'retrieve' %}<div class="card retrieval-shell">
<h2>🧠 Ask EchoMind</h2>
<p class="small">Ask about a saved object, place, person, or memory. Registered objects are shown first; only the same object appears in its history.</p>
<form method="POST" action="/retrieve"><textarea name="query" required placeholder="Where is my blue pen?">{{query}}</textarea><button>Find My Memory</button></form>

{% if result %}
{% if result.retrieval_mode == 'target_profile' and result.retrieval_match %}
<section class="primary-result-card profile-primary-card">
  <div class="section-kicker">🎯 PRIMARY RESULT · MY OBJECT</div>
  <h2 class="primary-title">{{result.retrieval_match.target_name}}</h2>
  <p class="section-subtitle">Your saved object record comes first. This picture, description, saved date/time, and location all belong to the same registered object.</p>
  <div class="primary-image-wrap profile-image-wrap">
    <div class="image-badge">⭐ EXACT REGISTERED OBJECT</div>
    <img class="primary-memory-image" src="{{result.registered_reference_image_url}}?v={{result.retrieval_match.target_id}}" alt="Registered object {{result.retrieval_match.target_name}}">
  </div>
  <div class="detail-section"><div class="detail-heading">📝 Your Saved Description</div><div class="detail-card manual-card">{{result.profile_description or 'No manual description was saved for this object.'}}</div></div>
  <div class="detail-section"><div class="detail-heading">🤖 AI Description</div><div class="detail-card ai-card">{{result.profile_ai_description or 'AI description is unavailable for the registered reference image.'}}</div></div>
  <div class="detail-section"><div class="detail-heading">🕒 Object Saved Date & Time</div><div class="datetime-grid">
    <div class="datetime-box"><div class="datetime-icon">📅</div><div class="datetime-label">DATE</div><div class="datetime-value">{{result.retrieval_match.profile_saved_date or 'Unavailable'}}</div></div>
    <div class="datetime-box"><div class="datetime-icon">📆</div><div class="datetime-label">DAY</div><div class="datetime-value">{{result.retrieval_match.profile_saved_day or 'Unavailable'}}</div></div>
    <div class="datetime-box"><div class="datetime-icon">🕒</div><div class="datetime-label">TIME</div><div class="datetime-value">{{result.retrieval_match.profile_saved_time or 'Unavailable'}}</div></div>
  </div></div>
  <div class="detail-section"><div class="detail-heading">📍 Saved Location</div><div class="context-grid">
    <div class="context-card"><div class="context-label">LOCATION</div><div class="context-value">{{result.retrieval_match.target_location or result.retrieval_match.target_room or 'Location not recorded'}}</div></div>
    <div class="context-card"><div class="context-label">OWNER</div><div class="context-value">{{result.retrieval_match.target_owner or 'Not specified'}}</div></div>
    <div class="context-card"><div class="context-label">CATEGORY</div><div class="context-value">{{result.retrieval_match.target_category or 'Not specified'}}</div></div>
  </div></div>
  <div class="primary-profile-status">🔒 <b>Identity protected:</b> Only this exact registered object is eligible for the memory history below. Generic or unrelated pictures are excluded.</div>
</section>

{% if results|length > 1 %}
<section class="history-section">
  <div class="section-kicker">📸 SAME-OBJECT MEMORY SIGHTINGS</div>
  <h2 class="history-title">Confirmed Pictures of {{result.retrieval_match.target_name}}</h2>
  <p class="section-subtitle">Every card below is a different saved memory of the same physical registered object. Each card keeps its own image, manual note, AI description, date/day/time, and location together.</p>
  {% for r in results[1:] %}
  <article class="history-card">
    <div class="history-number">{{loop.index}}</div><div class="history-content">
      {% if r.memory_image_url %}<img class="history-image" src="{{r.memory_image_url}}?v={{r.id}}" alt="Confirmed saved picture of {{result.retrieval_match.target_name}}">{% endif %}
      <div class="history-heading">📷 Confirmed Same Physical Object</div>
      <div class="history-detail"><div class="history-label">📝 Your Saved Description</div><div class="history-text">{{r.retrieval_saved_note or 'No manual description was saved for this memory.'}}</div></div>
      <div class="history-detail"><div class="history-label">🤖 AI Description</div><div class="history-text">{{r.retrieval_target_description or r.description or 'No AI description available.'}}</div></div>
      <div class="history-detail"><div class="history-label">🕒 Captured</div><div class="mini-datetime-grid">
        <div><b>📅 Date</b><span>{{r.matched_memory_date or r.date or 'Unavailable'}}</span></div><div><b>📆 Day</b><span>{{r.matched_memory_day or r.day or 'Unavailable'}}</span></div><div><b>🕒 Time</b><span>{{r.matched_memory_time or r.time or 'Unavailable'}}</span></div>
      </div></div>
      <div class="history-detail"><div class="history-label">📍 Location</div><div class="history-text">{{r.area or 'Location not recorded'}}</div>{% if r.latitude is not none and r.longitude is not none %}<div class="history-gps">GPS: {{'%.6f'|format(r.latitude)}}, {{'%.6f'|format(r.longitude)}}</div>{% endif %}</div>
    </div>
  </article>
  {% endfor %}
</section>

{% if results|length > 2 %}
<section class="roadmap-section">
  <div class="section-kicker">🗺️ MEMORY JOURNEY</div>
  <h2 class="history-title">{{result.retrieval_match.target_name}} — Timeline</h2>
  <p class="section-subtitle">Timeline mode is shown only because multiple confirmed memory pictures of the same object were found.</p>
  <div class="roadmap">
    <div class="roadmap-node primary-node"><div class="node-dot">⭐</div><div class="node-card primary-node-card"><div class="node-title">REGISTERED OBJECT</div><div class="node-meta">{{result.retrieval_match.profile_saved_day}} • {{result.retrieval_match.profile_saved_date}} • {{result.retrieval_match.profile_saved_time}}</div><div class="node-location">📍 {{result.retrieval_match.target_location or 'Location not recorded'}}</div></div></div>
    {% for r in results[1:] %}<div class="roadmap-node"><div class="node-dot secondary-node-dot">📷</div><div class="node-card"><div class="node-title">CONFIRMED MEMORY {{loop.index}}</div><div class="node-meta">{{r.matched_memory_day or r.day or 'Day unavailable'}} • {{r.matched_memory_date or r.date or 'Date unavailable'}} • {{r.matched_memory_time or r.time or 'Time unavailable'}}</div><div class="node-location">📍 {{r.area or 'Location not recorded'}}</div></div></div>{% endfor %}
  </div>
  {% set ns = namespace(has_map=false) %}{% for r in results[1:] %}{% if r.latitude is not none and r.longitude is not none %}{% set ns.has_map=true %}{% endif %}{% endfor %}
  {% if ns.has_map %}<div class="detail-section"><div class="detail-heading">🌍 Same-Object Location Map</div><div class="map-caption">Only confirmed memories of {{result.retrieval_match.target_name}} are plotted.</div><div class="object map-shell"><div id="memoryPathMap"></div></div></div>{% endif %}
</section>
<script>
(function(){const mapEl=document.getElementById('memoryPathMap');if(!mapEl||typeof L==='undefined')return;const points=[{% for r in results[1:] %}{% if r.latitude is not none and r.longitude is not none %}{lat:{{r.latitude}},lon:{{r.longitude}},title:{{('MEMORY ' ~ loop.index)|tojson}},place:{{(r.area or 'Location unavailable')|tojson}},date:{{((r.day or '') ~ ' • ' ~ (r.date or '') ~ ' • ' ~ (r.time or ''))|tojson}}}{% if not loop.last %},{% endif %}{% endif %}{% endfor %}];if(!points.length)return;const map=L.map(mapEl,{zoomControl:true,scrollWheelZoom:true});L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);const bounds=[],latlngs=[];points.forEach((p,i)=>{const icon=L.divIcon({className:'echomind-map-marker',html:'<div style="width:34px;height:34px;border-radius:50%;display:grid;place-items:center;background:#7c3aed;color:#fff;border:3px solid #fff;box-shadow:0 3px 12px rgba(0,0,0,.3);font-weight:900;font-size:13px;">'+(i+1)+'</div>',iconSize:[34,34],iconAnchor:[17,17],popupAnchor:[0,-18]});const marker=L.marker([p.lat,p.lon],{icon}).addTo(map);marker.bindPopup('<div style="min-width:190px"><strong>'+p.title+'</strong><br>'+p.place+'<br>'+p.date+'</div>');bounds.push([p.lat,p.lon]);latlngs.push([p.lat,p.lon]);});if(points.length>1)L.polyline(latlngs,{weight:5,opacity:.85}).addTo(map);if(bounds.length===1)map.setView(bounds[0],16);else map.fitBounds(bounds,{padding:[45,45],maxZoom:17});setTimeout(()=>map.invalidateSize(),250);})();
</script>
{% else %}
<div class="single-history-note">✨ One confirmed memory picture of this object is available. No timeline is shown until a second same-object memory is confirmed.</div>
{% endif %}
{% endif %}

{% else %}
<section class="primary-result-card">
  <div class="section-kicker">🔎 MEMORY RESULT</div><h2 class="primary-title">Memory Found</h2>
  {% if result.memory_image_url %}<img class="primary-memory-image" src="{{result.memory_image_url}}?v={{result.id}}" alt="Retrieved memory image">{% endif %}
  {% if result.user_description %}<div class="detail-section"><div class="detail-heading">📝 Your Saved Description</div><div class="detail-card manual-card">{{result.user_description}}</div></div>{% endif %}
  {% if result.description %}<div class="detail-section"><div class="detail-heading">🤖 AI Description</div><div class="detail-card ai-card">{{result.description}}</div></div>{% endif %}
  <div class="detail-section"><div class="detail-heading">🕒 When Was This Memory Captured?</div><div class="datetime-grid"><div class="datetime-box"><div class="datetime-icon">📅</div><div class="datetime-label">DATE</div><div class="datetime-value">{{result.date or 'Unavailable'}}</div></div><div class="datetime-box"><div class="datetime-icon">📆</div><div class="datetime-label">DAY</div><div class="datetime-value">{{result.day or 'Unavailable'}}</div></div><div class="datetime-box"><div class="datetime-icon">🕒</div><div class="datetime-label">TIME</div><div class="datetime-value">{{result.time or 'Unavailable'}}</div></div></div></div>
  <div class="detail-section"><div class="detail-heading">📍 Location & Context</div><div class="detail-card">{{result.area or 'Location not recorded'}}</div></div>
</section>
{% endif %}
{% else %}
{% if query %}<div class="error" style="margin-top:18px">{{answer}}</div>{% endif %}
{% endif %}
</div>{% endif %}

{% if page == 'database' %}<div class="card"><h2>🗄️ Persistent Memory Database</h2><p><b>{{memories|length}}</b> saved memories. SQLite is the authoritative metadata store; existing ChromaDB is never deleted automatically.</p>{% for m in memories %}<div class="object"><div class="two"><img class="memory-img" src="{{url_for('stored_memory_image',memory_id=m.id)}}" onerror="this.style.display='none';this.nextElementSibling.insertAdjacentHTML('afterbegin','<div class="error">Saved image could not be rendered.</div>')"><div><b>🗓️ Saved: {{memory_time_label(m)}}</b><br>Location: {{m.area}}<br><b>🧠 AI Description:</b> {{m.description}}{% if m.user_description %}<br><b>📝 User Note:</b> {{m.user_description}}{% endif %}<br>Objects: {% for o in m.objects %}{{o.name}}{% if not loop.last %}, {% endif %}{% endfor %}<br>Tags: {% for t in m.tags %}<span class="tag">{{t}}</span>{% endfor %}<br><br><a href="{{url_for('stored_memory_image',memory_id=m.id)}}" target="_blank">View stored image</a><form method="POST" action="{{url_for('delete_memory',memory_id=m.id)}}" style="margin-top:10px" onsubmit="return confirm('Delete this memory?')"><button class="danger">Delete Memory</button></form></div></div></div>{% else %}<p>No memories saved yet.</p>{% endfor %}</div>{% endif %}

</div>
<script>
let stream=null;
function toggleCustom(s,id){const el=document.getElementById(id); if(el){el.style.display=s.value==='__OTHER__'?'block':'none';el.required=s.value==='__OTHER__';}}
function localClockISO(){const d=new Date();const p=n=>String(n).padStart(2,'0');return d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate())+'T'+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());}
function stampUpload(){document.getElementById('upload_time').value=localClockISO();}
function stampProfileTime(form){const field=form.querySelector('input[name="browser_time"]');if(field){field.value=localClockISO();}}
async function startCamera(){try{if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error('Camera API unavailable');stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});document.getElementById('camera').srcObject=stream;document.getElementById('cameraStatus').innerText='✓ Camera permission granted.';}catch(e){document.getElementById('cameraStatus').innerText='Camera permission failed: '+e.message;}}
function captureImage(){const v=document.getElementById('camera'),c=document.getElementById('canvas');if(!v.videoWidth){alert('Click Allow Camera first and wait for the preview.');return;}c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);document.getElementById('image_data').value=c.toDataURL('image/jpeg',.96);document.getElementById('cam_time').value=localClockISO();document.getElementById('cameraStatus').innerText='Capturing GPS…';if(navigator.geolocation){navigator.geolocation.getCurrentPosition(p=>{document.getElementById('cam_lat').value=p.coords.latitude;document.getElementById('cam_lon').value=p.coords.longitude;document.getElementById('cam_acc').value=p.coords.accuracy||'';document.getElementById('cameraStatus').innerText='✓ Image + GPS captured. Saving memory…';document.getElementById('cameraForm').submit();},()=>{document.getElementById('cameraStatus').innerText='GPS unavailable. Saving image without GPS…';document.getElementById('cameraForm').submit();},{enableHighAccuracy:true,timeout:12000,maximumAge:0});}else{document.getElementById('cameraForm').submit();}}
function getUploadLocation(){if(!navigator.geolocation){document.getElementById('uploadStatus').innerText='GPS is not available.';return;}document.getElementById('uploadStatus').innerText='Requesting GPS…';navigator.geolocation.getCurrentPosition(p=>{document.getElementById('upload_latitude').value=p.coords.latitude;document.getElementById('upload_longitude').value=p.coords.longitude;document.getElementById('upload_accuracy').value=p.coords.accuracy||'';document.getElementById('uploadStatus').innerText='✓ GPS captured.'},e=>{document.getElementById('uploadStatus').innerText='GPS unavailable; the image can still be saved.'},{enableHighAccuracy:true,timeout:12000,maximumAge:0});}
</script></body></html>
'''


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 78)
    print("ECHOMIND IS READY")
    print("Open: http://127.0.0.1:5000")
    print("Persistent DB:", SQLITE_PATH)
    print("Uploads:", UPLOAD_DIR)
    print("Reference images:", REFERENCE_DIR)
    print("Models: YOLO11x / Qwen2.5-VL / BLIP fallback / CLIP / Sentence-BERT")
    print("=" * 78 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
