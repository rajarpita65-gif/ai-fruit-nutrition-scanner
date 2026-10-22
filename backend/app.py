import os
import json
import io
import threading
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(PROJECT_ROOT, ".yolo"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, ".cache"))
os.environ.setdefault("HOME", PROJECT_ROOT)
os.environ.setdefault("USERPROFILE", PROJECT_ROOT)

try:
    import torch
    import open_clip
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    open_clip = None
try:
    from ultralytics import YOLOWorld
except Exception:  # pragma: no cover - optional runtime dependency
    YOLOWorld = None
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOADS_DIR, exist_ok=True)

# ─── Globals ──────────────────────────────────────────────────────────────────
image_model        = None   # MobileNetV2 image classifier (optional)
image_model_source = "none" # "custom" | "imagenet" | "none"
label_map          = None   # {str_index: class_name}
calorie_model      = None   # Calorie class classifier
calorie_labels     = None   # {"0":"low","1":"medium","2":"high"}
nutrition_df       = None   # nutrition.csv loaded as DataFrame
_calorie_ready     = False  # True once background warm-up finishes
_calorie_lock      = threading.Lock()
zero_shot_model         = None
zero_shot_preprocess    = None
zero_shot_tokenizer     = None
zero_shot_text_features = None
zero_shot_candidate_labels = []
zero_shot_device        = "cpu"
yolo_world_model        = None
yolo_world_labels       = []

IMAGENET_TO_FOOD = {
    "banana": "Banana",
    "bell_pepper": "Bell Pepper",
    "cucumber": "Cucumber",
    "ear": "Corn",
    "fig": "Fig",
    "granny_smith": "Apple Granny Smith",
    "jackfruit": "Jackfruit",
    "lemon": "Lemon",
    "orange": "Orange",
    "pineapple": "Pineapple",
    "pomegranate": "Pomegranate",
    "strawberry": "Strawberry",
    "zucchini": "Zucchini",
}

ZERO_SHOT_PROMPTS = {
    "Apple": "a close-up photo of an apple",
    "Apricot": "a close-up photo of an apricot",
    "Avocado": "a close-up photo of an avocado",
    "Banana": "a close-up photo of a banana",
    "Bell Pepper": "a close-up photo of a bell pepper",
    "Blackberry": "a close-up photo of blackberries",
    "Blueberry": "a close-up photo of blueberries",
    "Cantaloupe": "a close-up photo of a cantaloupe melon",
    "Cherry": "a close-up photo of cherries",
    "Clementine": "a close-up photo of a clementine",
    "Coconut": "a close-up photo of a coconut",
    "Corn": "a close-up photo of corn on the cob",
    "Cucumber": "a close-up photo of a cucumber",
    "Dragon Fruit": "a close-up photo of a dragon fruit",
    "Fig": "a close-up photo of a fig",
    "Grape": "a close-up photo of grapes",
    "Grape White": "a close-up photo of a bunch of green grapes",
    "Grape Pink": "a close-up photo of a bunch of red grapes",
    "Grape Blue": "a close-up photo of a bunch of purple grapes",
    "Grapefruit": "a close-up photo of a grapefruit",
    "Guava": "a close-up photo of a guava",
    "Jackfruit": "a close-up photo of a jackfruit",
    "Kiwi": "a close-up photo of a kiwi fruit",
    "Lemon": "a close-up photo of a lemon",
    "Lime": "a close-up photo of a lime",
    "Longan": "a close-up photo of longan fruit",
    "Lychee": "a close-up photo of lychee fruit",
    "Mango": "a close-up photo of a mango",
    "Mangostan": "a close-up photo of mangosteen fruit",
    "Mulberry": "a close-up photo of mulberries",
    "Nectarine": "a close-up photo of a nectarine",
    "Orange": "a close-up photo of an orange",
    "Papaya": "a close-up photo of a papaya",
    "Passion Fruit": "a close-up photo of a passion fruit",
    "Peach": "a close-up photo of a peach",
    "Pear": "a close-up photo of a pear",
    "Persimmon": "a close-up photo of a persimmon",
    "Physalis": "a close-up photo of physalis fruit",
    "Pineapple": "a close-up photo of a pineapple",
    "Pitahaya Red": "a close-up photo of a red dragon fruit",
    "Plum": "a close-up photo of a plum",
    "Pomegranate": "a close-up photo of a pomegranate",
    "Pomelo": "a close-up photo of a pomelo",
    "Quince": "a close-up photo of a quince",
    "Rambutan": "a close-up photo of a rambutan",
    "Raspberry": "a close-up photo of raspberries",
    "Strawberry": "a close-up photo of a strawberry",
    "Tamarillo": "a close-up photo of a tamarillo",
    "Tangerine": "a close-up photo of a tangerine",
    "Tomato": "a close-up photo of a tomato",
    "Watermelon": "a close-up photo of a watermelon slice",
    "Zucchini": "a close-up photo of a zucchini",
}

YOLO_WORLD_LABEL_MAP = {
    "apple": "Apple",
    "avocado": "Avocado",
    "banana": "Banana",
    "bell pepper": "Bell Pepper",
    "coconut": "Coconut",
    "cucumber": "Cucumber",
    "grapes": "Grape",
    "green grapes": "Grape White",
    "guava": "Guava",
    "kiwi": "Kiwi",
    "lemon": "Lemon",
    "lime": "Lime",
    "mango": "Mango",
    "orange": "Orange",
    "papaya": "Papaya",
    "peach": "Peach",
    "pear": "Pear",
    "pineapple": "Pineapple",
    "pomegranate": "Pomegranate",
    "red grapes": "Grape Pink",
    "strawberry": "Strawberry",
    "tomato": "Tomato",
    "watermelon": "Watermelon",
}


# ─── Background warm-up ───────────────────────────────────────────────────────
def _warmup_calorie():
    """Run one dummy inference so TF compiles the graph before any user request."""
    global _calorie_ready
    if calorie_model is None:
        return
    try:
        dummy = np.zeros((1, 5), dtype=np.float32)
        calorie_model(dummy, training=False)   # __call__ is faster than .predict()
        with _calorie_lock:
            _calorie_ready = True
        print("[OK] Calorie model warm-up complete -- ready for requests")
    except Exception as exc:
        print(f"[WARN] Warm-up failed: {exc}")


def canonical_food_label(food_name: str) -> str:
    lower = food_name.lower().strip()
    if lower.startswith("bell pepper"):
        return "Bell Pepper"
    if lower.startswith("passion fruit"):
        return "Passion Fruit"
    if lower.startswith("pitahaya"):
        return "Pitahaya Red"
    if lower.startswith("grape white"):
        return "Grape White"
    if lower.startswith("grape blue"):
        return "Grape Blue"
    if lower.startswith("grape pink"):
        return "Grape Pink"
    if lower.startswith("grape"):
        return "Grape"
    if lower.startswith("apple"):
        return "Apple"
    if lower.startswith("banana"):
        return "Banana"
    if lower.startswith("pear"):
        return "Pear"
    if lower.startswith("mango"):
        return "Mango"
    if lower.startswith("orange"):
        return "Orange"
    if lower.startswith("peach"):
        return "Peach"
    if lower.startswith("pineapple"):
        return "Pineapple"
    if lower.startswith("watermelon"):
        return "Watermelon"
    if lower.startswith("tomato"):
        return "Tomato"
    return " ".join(part.capitalize() for part in lower.split())


def build_zero_shot_candidates() -> list[str]:
    if nutrition_df is None:
        return []
    labels = {
        canonical_food_label(str(food))
        for food in nutrition_df["food"].dropna().astype(str)
    }
    return sorted(labels)


def zero_shot_prompt(label: str) -> str:
    return ZERO_SHOT_PROMPTS.get(label, f"a close-up photo of {label.lower()}")


def load_zero_shot_model() -> bool:
    global image_model, zero_shot_model, zero_shot_preprocess, zero_shot_tokenizer
    global zero_shot_text_features, zero_shot_candidate_labels, zero_shot_device

    if torch is None or open_clip is None or nutrition_df is None:
        return False

    try:
        zero_shot_device = "cuda" if torch.cuda.is_available() else "cpu"
        zero_shot_model, _, zero_shot_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )
        zero_shot_model = zero_shot_model.to(zero_shot_device)
        zero_shot_model.eval()
        zero_shot_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        zero_shot_candidate_labels = build_zero_shot_candidates()

        prompts = [zero_shot_prompt(label) for label in zero_shot_candidate_labels]
        with torch.no_grad():
            text_tokens = zero_shot_tokenizer(prompts).to(zero_shot_device)
            zero_shot_text_features = zero_shot_model.encode_text(text_tokens)
            zero_shot_text_features /= zero_shot_text_features.norm(dim=-1, keepdim=True)

        image_model = zero_shot_model
        return True
    except Exception as exc:
        zero_shot_model = None
        zero_shot_preprocess = None
        zero_shot_tokenizer = None
        zero_shot_text_features = None
        zero_shot_candidate_labels = []
        print(f"[WARN] OpenCLIP zero-shot load failed: {exc}")
        return False


def load_yolo_world_model() -> bool:
    global yolo_world_model, yolo_world_labels

    if YOLOWorld is None:
        return False

    try:
        os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
        os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

        candidate_paths = [
            os.path.join(MODEL_DIR, "yolov8s-world.pt"),
            os.path.join(PROJECT_ROOT, "yolov8s-world.pt"),
        ]
        model_source = next((path for path in candidate_paths if os.path.exists(path)), "yolov8s-world.pt")
        yolo_world_model = YOLOWorld(model_source)
        yolo_world_labels = list(YOLO_WORLD_LABEL_MAP.keys())
        yolo_world_model.set_classes(yolo_world_labels)
        return True
    except Exception as exc:
        yolo_world_model = None
        yolo_world_labels = []
        print(f"[WARN] YOLO-World load failed: {exc}")
        return False


# ─── Resource loader (runs once at startup) ───────────────────────────────────
def load_resources():
    global image_model, image_model_source, label_map, calorie_model, calorie_labels, nutrition_df

    # Nutrition CSV
    nutrition_path = os.path.join(DATASET_DIR, "nutrition.csv")
    if os.path.exists(nutrition_path):
        try:
            nutrition_df = pd.read_csv(nutrition_path)
            nutrition_df.columns = nutrition_df.columns.str.lower().str.strip()
            print(f"[OK] Nutrition CSV loaded: {len(nutrition_df)} items")
        except Exception as exc:
            print(f"[WARN] Nutrition CSV load failed: {exc}")

    if load_yolo_world_model():
        print("[OK] YOLO-World detector loaded for fruit localization")
    else:
        print("[WARN] YOLO-World detector unavailable -- detection will use full image analysis")

    # Image model (optional -- needs Fruits-360 training)
    for path in [
        os.path.join(MODEL_DIR, "nutrition_model.keras"),
        os.path.join(MODEL_DIR, "nutrition_model.h5"),
    ]:
        if os.path.exists(path):
            try:
                image_model = tf.keras.models.load_model(path)
                image_model_source = "custom"
                print(f"[OK] Image model loaded: {path}")
            except Exception as exc:
                print(f"[WARN] Image model load failed: {exc}")
            break
    if not image_model:
        if load_zero_shot_model():
            image_model_source = "openclip-zero-shot"
            print("[OK] OpenCLIP zero-shot model loaded for fruit recognition")
        else:
            try:
                image_model = MobileNetV2(weights="imagenet")
                image_model_source = "imagenet"
                print("[OK] MobileNetV2 ImageNet model loaded for fruit recognition")
            except Exception as exc:
                image_model = None
                image_model_source = "none"
                print(f"[WARN] ImageNet model unavailable: {exc}")
                print("[INFO] Falling back to heuristic demo mode")

    # Label map
    label_map_path = os.path.join(MODEL_DIR, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            label_map = json.load(f)
        print(f"[OK] Label map: {len(label_map)} classes")

    # Calorie classifier
    calorie_model_path = os.path.join(MODEL_DIR, "calorie_classifier.keras")
    if os.path.exists(calorie_model_path):
        try:
            calorie_model = tf.keras.models.load_model(calorie_model_path)
            print("[OK] Calorie classifier loaded -- warming up in background...")
            # Start non-blocking warm-up so Flask accepts requests immediately
            threading.Thread(target=_warmup_calorie, daemon=True).start()
        except Exception as exc:
            print(f"[WARN] Calorie model load failed: {exc}")
    else:
        print("[INFO] calorie_classifier.keras not found -- run train_model.py")

    calorie_labels_path = os.path.join(MODEL_DIR, "calorie_class_labels.json")
    if os.path.exists(calorie_labels_path):
        with open(calorie_labels_path) as f:
            calorie_labels = json.load(f)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes, model_source: str) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    batch = np.expand_dims(np.array(img, dtype=np.float32), 0)
    if model_source == "imagenet":
        return preprocess_input(batch.copy())
    return batch / 255.0


def normalize_label(name: str) -> str:
    return " ".join(p for p in name.lower().strip().split() if not p.isdigit())


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def get_col(row, *keys):
    for k in keys:
        if k in row.index and pd.notna(row[k]):
            try:
                return float(row[k])
            except (ValueError, TypeError):
                pass
    return 0.0


def extract_nutrition_row(row) -> dict:
    return {
        "calories":      get_col(row, "calories", "calorie", "energy"),
        "protein":       get_col(row, "protein", "proteins"),
        "carbohydrates": get_col(row, "carbohydrates", "carbs", "carbohydrate"),
        "fat":           get_col(row, "fat", "total fat", "fats"),
        "fiber":         get_col(row, "fiber", "dietary fiber", "fibre"),
        "sugar":         get_col(row, "sugar", "sugars"),
    }


def find_nutrition(food_name: str) -> dict | None:
    if nutrition_df is None:
        return None
    name     = normalize_label(food_name)
    food_col = nutrition_df["food"].str.lower().str.strip()
    first    = name.split()[0] if name.split() else name

    for mask in [food_col == name, food_col == first,
                 food_col.str.contains(first, na=False, regex=False)]:
        if mask.any():
            return extract_nutrition_row(nutrition_df[mask].iloc[0])

    for _, row in nutrition_df.iterrows():
        if str(row.get("food", "")).lower().strip() in name:
            return extract_nutrition_row(row)
    return None


def calculate_nutrition(per_100g: dict, weight_g: float) -> dict:
    f = weight_g / 100.0
    return {k: round(v * f, 2) for k, v in per_100g.items()}


def find_food_name_match(text: str) -> str | None:
    if nutrition_df is None or not text:
        return None

    normalized = normalize_label(
        text.replace("_", " ").replace("-", " ").replace(".", " ")
    )
    if not normalized:
        return None

    food_col = nutrition_df["food"].astype(str).str.lower().str.strip()

    exact = nutrition_df[food_col == normalized]
    if not exact.empty:
        return str(exact.iloc[0]["food"]).title()

    for token in normalized.split():
        if len(token) < 3:
            continue
        token_match = nutrition_df[food_col == token]
        if not token_match.empty:
            return str(token_match.iloc[0]["food"]).title()

    for token in normalized.split():
        if len(token) < 3:
            continue
        token_match = nutrition_df[food_col.str.contains(token, na=False, regex=False)]
        if not token_match.empty:
            return str(token_match.iloc[0]["food"]).title()

    return None


def detect_grape_cluster(image_rgb: np.ndarray) -> tuple[str, float, str] | None:
    small = cv2.resize(image_rgb, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

    green_mask = cv2.inRange(hsv, (28, 35, 45), (95, 255, 255))
    purple_mask = cv2.inRange(hsv, (110, 35, 20), (170, 255, 220))

    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)

    def grape_peaks(mask: np.ndarray) -> tuple[int, float]:
        coverage = float(np.count_nonzero(mask)) / mask.size
        if coverage < 0.05:
            return 0, coverage
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        peak_kernel = np.ones((7, 7), np.float32)
        dilated = cv2.dilate(dist, peak_kernel)
        peaks = ((dist == dilated) & (dist > 4.5)).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(peaks)
        peak_count = 0
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            if 1 <= area <= 20:
                peak_count += 1
        return peak_count, coverage

    green_peaks, green_coverage = grape_peaks(green_mask)
    purple_peaks, purple_coverage = grape_peaks(purple_mask)

    if green_peaks >= 3 and green_coverage > 0.10:
        return "Grape White", min(92.0, 72.0 + green_peaks * 2.2), "cluster"
    if purple_peaks >= 3 and purple_coverage > 0.08:
        return "Grape Blue", min(92.0, 72.0 + purple_peaks * 2.2), "cluster"
    return None


def guess_food_from_image(image_bytes: bytes, filename: str = "") -> tuple[str, float, str]:
    name_guess = find_food_name_match(os.path.splitext(filename)[0])
    if name_guess:
        return name_guess, 96.0, "filename"

    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb_full = np.asarray(base_img, dtype=np.uint8)

        grape_cluster = detect_grape_cluster(rgb_full)
        if grape_cluster is not None:
            return grape_cluster

        img = base_img.resize((64, 64))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        center = arr[8:56, 8:56]
        channel_max = center.max(axis=-1)
        channel_min = center.min(axis=-1)
        saturation = channel_max - channel_min
        foreground_mask = (saturation > 0.10) | (channel_max < 0.92)

        sample = center[foreground_mask] if np.count_nonzero(foreground_mask) > 200 else center.reshape(-1, 3)
        red = float(sample[:, 0].mean())
        green = float(sample[:, 1].mean())
        blue = float(sample[:, 2].mean())
        brightness = float(sample.mean())

        hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
        hsv_center = hsv[8:56, 8:56]
        hsv_sample = hsv_center[foreground_mask] if np.count_nonzero(foreground_mask) > 200 else hsv_center.reshape(-1, 3)
        hue = hsv_sample[:, 0]

        red_ratio = float(np.mean((hue <= 12) | (hue >= 245)))
        orange_ratio = float(np.mean((hue > 12) & (hue <= 28)))
        yellow_ratio = float(np.mean((hue > 28) & (hue <= 48)))
        green_ratio = float(np.mean((hue > 48) & (hue <= 105)))

        ys, xs = np.where(foreground_mask)
        if len(xs) and len(ys):
            bbox_w = max(xs.max() - xs.min() + 1, 1)
            bbox_h = max(ys.max() - ys.min() + 1, 1)
            aspect = bbox_w / bbox_h
        else:
            aspect = 1.0

        yellow_orange = yellow_ratio + orange_ratio

        if yellow_ratio > 0.35 and red_ratio > 0.18 and green_ratio < 0.22 and blue < 0.32:
            return "Mango", 86.0, "color"
        if yellow_orange > 0.34 and red_ratio > 0.10 and aspect > 1.00:
            return "Mango", 84.0, "color"
        if yellow_ratio > 0.42 and aspect > 1.25:
            return "Banana", 80.0, "color"
        if orange_ratio > 0.28 and yellow_ratio > 0.12 and aspect > 1.05:
            return "Mango", 77.0, "color"
        if orange_ratio > 0.30 and aspect <= 1.15:
            return "Orange", 76.0, "color"
        if yellow_ratio > 0.36 and green_ratio < 0.16:
            return "Lemon", 74.0, "color"
        if red_ratio > 0.22 and yellow_orange > 0.18 and aspect > 1.05:
            return "Mango", 72.0, "color"

        if red > 0.62 and green > 0.56 and blue < 0.42:
            return "Banana", 78.0, "color"
        if red > 0.72 and green > 0.35 and blue < 0.24:
            return "Orange", 76.0, "color"
        if red > 0.58 and green < 0.48 and blue < 0.48:
            return "Apple", 74.0, "color"
        if green > red and green > blue and brightness > 0.40:
            return "Pear", 72.0, "color"
        if green > 0.34 and blue < 0.28:
            return "Kiwi", 68.0, "color"
        if blue > red and blue > green:
            return "Blueberry", 66.0, "color"
    except Exception as exc:
        print(f"[WARN] Demo image fallback failed: {exc}")

    return "Apple", 60.0, "fallback"


def map_imagenet_label_to_food(label: str) -> str | None:
    mapped = IMAGENET_TO_FOOD.get(label)
    if mapped:
        return mapped

    normalized = label.replace("_", " ").strip()
    if normalized:
        return find_food_name_match(normalized)
    return None


def same_food_family(name_a: str, name_b: str) -> bool:
    return canonical_food_label(name_a) == canonical_food_label(name_b)


def map_yolo_label_to_food(label: str) -> str | None:
    if not label:
        return None

    normalized = label.lower().strip()
    mapped = YOLO_WORLD_LABEL_MAP.get(normalized)
    if mapped:
        return mapped

    return find_food_name_match(normalized)


def detect_fruit_region_with_yolo(image_bytes: bytes) -> dict | None:
    if yolo_world_model is None:
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        results = yolo_world_model.predict(image, conf=0.05, imgsz=640, verbose=False)
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        image_area = max(width * height, 1)
        best_detection = None

        for index, bbox in enumerate(xyxy):
            class_id = class_ids[index]
            if class_id >= len(yolo_world_labels):
                continue

            raw_label = yolo_world_labels[class_id]
            food_name = map_yolo_label_to_food(raw_label)
            if not food_name:
                continue

            x1, y1, x2, y2 = [int(round(value)) for value in bbox.tolist()]
            box_width = max(x2 - x1, 1)
            box_height = max(y2 - y1, 1)
            area_ratio = (box_width * box_height) / image_area
            score = float(confs[index]) * (1.0 + min(area_ratio, 0.8))

            if best_detection is None or score > best_detection["score"]:
                best_detection = {
                    "raw_label": raw_label,
                    "food_name": food_name,
                    "confidence": float(confs[index]) * 100.0,
                    "bbox": [x1, y1, x2, y2],
                    "area_ratio": area_ratio,
                    "score": score,
                }

        if best_detection is None:
            return None

        x1, y1, x2, y2 = best_detection["bbox"]
        pad_x = max(int((x2 - x1) * 0.12), 8)
        pad_y = max(int((y2 - y1) * 0.12), 8)
        crop_box = (
            max(x1 - pad_x, 0),
            max(y1 - pad_y, 0),
            min(x2 + pad_x, width),
            min(y2 + pad_y, height),
        )
        crop = image.crop(crop_box)
        if crop.size[0] < 24 or crop.size[1] < 24:
            return None

        return {
            "food_name": best_detection["food_name"],
            "raw_label": best_detection["raw_label"],
            "confidence": round(best_detection["confidence"], 2),
            "bbox": best_detection["bbox"],
            "area_ratio": round(best_detection["area_ratio"], 4),
            "crop_bytes": image_to_jpeg_bytes(crop),
        }
    except Exception as exc:
        print(f"[WARN] YOLO crop failed: {exc}")
        return None


def is_camera_fruit_detection_valid(detection: dict | None) -> bool:
    if detection is None:
        return False
    return (
        float(detection.get("confidence", 0.0)) >= 20.0
        and float(detection.get("area_ratio", 0.0)) >= 0.015
    )


def image_variants_for_zero_shot(image: Image.Image) -> list[Image.Image]:
    variants = [image]
    w, h = image.size
    short = min(w, h)

    if short > 0:
        left = max((w - short) // 2, 0)
        top = max((h - short) // 2, 0)
        center_square = image.crop((left, top, left + short, top + short))
        variants.append(center_square)

        tighter = max(int(short * 0.78), 32)
        left_tight = max((w - tighter) // 2, 0)
        top_tight = max((h - tighter) // 2, 0)
        variants.append(image.crop((left_tight, top_tight, left_tight + tighter, top_tight + tighter)))

    return variants


def predict_food_with_zero_shot(image_bytes: bytes) -> tuple[str | None, float, list[tuple[str, float]]]:
    if zero_shot_model is None or zero_shot_text_features is None or not zero_shot_candidate_labels:
        return None, 0.0, []

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    variants = image_variants_for_zero_shot(image)

    with torch.no_grad():
        batch = torch.stack([zero_shot_preprocess(img) for img in variants]).to(zero_shot_device)
        image_features = zero_shot_model.encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        pooled = image_features.mean(dim=0, keepdim=True)
        pooled /= pooled.norm(dim=-1, keepdim=True)

        logits = 100.0 * pooled @ zero_shot_text_features.T
        probs = logits.softmax(dim=-1)[0]
        topk = probs.topk(min(5, len(zero_shot_candidate_labels)))

    top_candidates = [
        (zero_shot_candidate_labels[idx], float(score) * 100.0)
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist())
    ]
    if not top_candidates:
        return None, 0.0, []

    best_label, best_confidence = top_candidates[0]
    return best_label, best_confidence, top_candidates


def predict_food_with_imagenet(predictions: np.ndarray) -> tuple[str | None, float, str | None]:
    decoded = decode_predictions(predictions, top=5)[0]
    best_name = None
    best_confidence = 0.0
    best_label = None

    for _, label, score in decoded:
        mapped_name = map_imagenet_label_to_food(label)
        if mapped_name and float(score) > best_confidence:
            best_name = mapped_name
            best_confidence = float(score)
            best_label = label

    if best_name:
        return best_name, round(best_confidence * 100, 2), best_label
    return None, 0.0, None


def predict_calorie_class(nutrition: dict) -> str:
    """
    Use trained classifier when warmed up, heuristic otherwise.
    Avoids blocking requests during the ~15 s TF JIT warm-up window.
    """
    cal = nutrition.get("calories", 0)

    with _calorie_lock:
        ready = _calorie_ready

    if not ready or calorie_model is None or calorie_labels is None:
        # Fast heuristic while model is warming up
        if cal < 150:  return "low"
        if cal < 400:  return "medium"
        return "high"

    try:
        features = np.array([[
            nutrition.get("protein", 0),
            nutrition.get("carbohydrates", 0),
            nutrition.get("fat", 0),
            nutrition.get("fiber", 0),
            nutrition.get("sugar", 0),
        ]], dtype=np.float32)
        # Use __call__ (faster than .predict for single samples — no batch overhead)
        pred    = calorie_model(features, training=False).numpy()[0]
        top_idx = str(int(np.argmax(pred)))
        return calorie_labels.get(top_idx, "medium")
    except Exception as exc:
        print(f"[WARN] calorie prediction failed: {exc}")
        if cal < 150:  return "low"
        if cal < 400:  return "medium"
        return "high"


def demo_predict(image_bytes: bytes, filename: str, weight: float):
    """Return a best-effort fruit guess when no image model is available."""
    if nutrition_df is None:
        return None

    yolo_detection = detect_fruit_region_with_yolo(image_bytes)
    if yolo_detection is not None:
        nutrition_per_100g = find_nutrition(yolo_detection["food_name"])
        if nutrition_per_100g is not None:
            return build_success_result(
                yolo_detection["food_name"],
                yolo_detection["confidence"],
                weight,
                nutrition_per_100g=nutrition_per_100g,
                detected_by=f"yolo:{yolo_detection['raw_label']}",
                recognition_model="yoloworld-detector",
                message="YOLO localized the fruit before nutrition analysis.",
            )

    name, confidence, detected_by = guess_food_from_image(image_bytes, filename)
    row_match = nutrition_df[
        nutrition_df["food"].astype(str).str.lower().str.strip() == name.lower()
    ]
    row = row_match.iloc[0] if not row_match.empty else nutrition_df.iloc[0]
    name = str(row.get("food", name)).title()
    per  = extract_nutrition_row(row)
    return build_success_result(
        name,
        confidence,
        weight,
        nutrition_per_100g=per,
        detected_by=detected_by,
        recognition_model="heuristic-fallback",
        demo_mode=True,
    )


def build_success_result(food_name: str, confidence: float, weight: float, *,
                         nutrition_per_100g: dict, detected_by: str | None = None,
                         recognition_model: str | None = None, message: str | None = None,
                         demo_mode: bool = False,
                         generic_model_prediction: str | None = None,
                         generic_model_confidence: float | None = None) -> dict:
    nutrition_calc = calculate_nutrition(nutrition_per_100g, weight)
    result = {
        "status":             "success",
        "food_name":          food_name,
        "confidence":         round(confidence, 2),
        "weight":             weight,
        "nutrition":          nutrition_calc,
        "nutrition_per_100g": nutrition_per_100g,
        "calorie_class":      predict_calorie_class(nutrition_calc),
    }
    if detected_by:
        result["detected_by"] = detected_by
    if recognition_model:
        result["recognition_model"] = recognition_model
    if message:
        result["message"] = message
    if generic_model_prediction:
        result["generic_model_prediction"] = generic_model_prediction
    if generic_model_confidence is not None:
        result["generic_model_confidence"] = round(generic_model_confidence, 2)
    if demo_mode:
        result["demo_mode"] = True
        result["model_warming_up"] = not _calorie_ready
    return result


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":    "AI Edge Nutrition Scanner API",
        "status":  "running",
        "version": "1.0.0",
        "image_model_source": image_model_source,
        "calorie_model_ready": _calorie_ready,
        "endpoints": {
            "GET  /health":  "Server and model status",
            "POST /predict": "Predict food from image + weight",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":                "ok",
        "image_model_loaded":    image_model is not None,
        "image_model_source":    image_model_source,
        "detector_model_loaded": yolo_world_model is not None,
        "detector_model_source": "yoloworld" if yolo_world_model is not None else "none",
        "calorie_model_loaded":  calorie_model is not None,
        "calorie_model_ready":   _calorie_ready,
        "nutrition_data_loaded": nutrition_df is not None,
        "classes":               len(label_map) if label_map else 0,
    })


@app.route("/predict", methods=["POST"])
def predict():
    # Validate weight
    weight_str = request.form.get("weight", "").strip()
    source = request.form.get("source", "").strip().lower()
    is_camera_source = source == "camera"
    if not weight_str:
        return jsonify({"status": "error", "message": "Weight is required."}), 400
    try:
        weight = float(weight_str)
        if weight <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"status": "error", "message": "Enter a positive weight in grams."}), 400

    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    yolo_detection = detect_fruit_region_with_yolo(image_bytes)
    if is_camera_source and not is_camera_fruit_detection_valid(yolo_detection):
        return jsonify({
            "status": "no_detection",
            "message": "No fruit detected in the live camera frame. Hold one fruit clearly in front of the camera.",
        })
    analysis_bytes = yolo_detection["crop_bytes"] if yolo_detection else image_bytes
    detection_model = f"yolo:{yolo_detection['raw_label']}" if yolo_detection else None

    # ── Demo mode ─────────────────────────────────────────────────────────────
    if image_model is None:
        result = demo_predict(image_bytes, image_file.filename or "", weight)
        if result:
            return jsonify(result)
        return jsonify({
            "status":  "error",
            "message": "Nutrition data not found. Ensure nutrition.csv is in dataset/ folder.",
        }), 500

    if image_model_source == "openclip-zero-shot":
        heuristic_name, heuristic_confidence, heuristic_source = guess_food_from_image(
            image_bytes,
            image_file.filename or "",
        )
        food_name, confidence, top_candidates = predict_food_with_zero_shot(analysis_bytes)

        if food_name is None:
            if yolo_detection is not None:
                nutrition_per_100g = find_nutrition(yolo_detection["food_name"])
                if nutrition_per_100g is not None:
                    return jsonify(build_success_result(
                        yolo_detection["food_name"],
                        yolo_detection["confidence"],
                        weight,
                        nutrition_per_100g=nutrition_per_100g,
                        detected_by=detection_model,
                        recognition_model="yoloworld-detector",
                        message="Used YOLO fruit localization because the classifier could not rank this image confidently.",
                    ))
            result = demo_predict(image_bytes, image_file.filename or "", weight)
            if result:
                result["message"] = "Used fruit-specific visual fallback because the zero-shot model could not rank this image."
                return jsonify(result)
            return jsonify({
                "status": "error",
                "message": "Could not recognize a supported fruit or food from this image.",
            }), 422

        if yolo_detection is not None:
            yolo_name = yolo_detection["food_name"]
            yolo_confidence = yolo_detection["confidence"]

            if same_food_family(food_name, yolo_name):
                resolved_name = food_name if find_nutrition(food_name) is not None else yolo_name
                nutrition_per_100g = find_nutrition(resolved_name)
                if nutrition_per_100g is not None:
                    combined_confidence = min(
                        98.0,
                        max(confidence, yolo_confidence, (confidence * 0.72) + (yolo_confidence * 0.35)),
                    )
                    return jsonify(build_success_result(
                        resolved_name,
                        combined_confidence,
                        weight,
                        nutrition_per_100g=nutrition_per_100g,
                        detected_by=detection_model,
                        recognition_model="yoloworld+openclip",
                        message="YOLO localized the fruit before OpenCLIP classified it.",
                    ))

            if yolo_confidence >= 45 and confidence < 68:
                nutrition_per_100g = find_nutrition(yolo_name)
                if nutrition_per_100g is not None:
                    top_prediction = top_candidates[0] if top_candidates else (None, 0.0)
                    return jsonify(build_success_result(
                        yolo_name,
                        max(yolo_confidence, confidence),
                        weight,
                        nutrition_per_100g=nutrition_per_100g,
                        detected_by=detection_model,
                        recognition_model="yoloworld-detector",
                        message="Used YOLO fruit localization because the classifier disagreed with low confidence.",
                        generic_model_prediction=top_prediction[0],
                        generic_model_confidence=top_prediction[1],
                    ))

        if (
            yolo_detection is None
            and heuristic_name
            and heuristic_confidence >= 76
            and (
                confidence < 45
                or not same_food_family(food_name, heuristic_name)
            )
        ):
            heuristic_nutrition = find_nutrition(heuristic_name)
            if heuristic_nutrition is not None:
                top_prediction = top_candidates[0] if top_candidates else (None, 0.0)
                return jsonify(build_success_result(
                    heuristic_name,
                    heuristic_confidence,
                    weight,
                    nutrition_per_100g=heuristic_nutrition,
                    detected_by=heuristic_source,
                    recognition_model="hybrid-heuristic",
                    message="Used fruit-specific visual fallback because the zero-shot model was uncertain.",
                    demo_mode=True,
                    generic_model_prediction=top_prediction[0],
                    generic_model_confidence=top_prediction[1],
                ))

        nutrition_per_100g = find_nutrition(food_name)
        if nutrition_per_100g is None:
            return jsonify({
                "status":     "not_found",
                "food_name":  food_name,
                "confidence": round(confidence, 2),
                "message":    f"Nutrition data not found for '{food_name}'.",
            })

        return jsonify(build_success_result(
            food_name,
            confidence,
            weight,
            nutrition_per_100g=nutrition_per_100g,
            detected_by=detection_model,
            recognition_model="yoloworld+openclip" if yolo_detection else "openclip-zero-shot",
            message="YOLO localized the fruit before classification." if yolo_detection else None,
        ))

    # ── Real image prediction ─────────────────────────────────────────────────
    try:
        img_array = preprocess_image(analysis_bytes, image_model_source)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Cannot read image: {exc}"}), 400

    predictions   = image_model.predict(img_array, verbose=0)
    if image_model_source == "custom":
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx]) * 100

        str_key  = str(predicted_idx)
        food_raw = label_map.get(predicted_idx) or label_map.get(str_key, "Unknown")
        if isinstance(food_raw, int):
            food_raw = {v: k for k, v in label_map.items()}.get(predicted_idx, "Unknown")

        food_name = normalize_label(food_raw).title()

        if confidence < 70:
            if yolo_detection is not None:
                nutrition_per_100g = find_nutrition(yolo_detection["food_name"])
                if nutrition_per_100g is not None:
                    return jsonify(build_success_result(
                        yolo_detection["food_name"],
                        max(yolo_detection["confidence"], confidence),
                        weight,
                        nutrition_per_100g=nutrition_per_100g,
                        detected_by=detection_model,
                        recognition_model="yoloworld+custom",
                        message="Used YOLO fruit localization because the custom classifier confidence was low.",
                    ))
            return jsonify({
                "status":     "low_confidence",
                "food_name":  food_name,
                "confidence": round(confidence, 2),
                "message":    "Low confidence. Please use a clearer image.",
            })
    else:
        heuristic_name, heuristic_confidence, heuristic_source = guess_food_from_image(
            image_bytes,
            image_file.filename or "",
        )
        food_name, confidence, raw_label = predict_food_with_imagenet(predictions)
        if food_name is None:
            if yolo_detection is not None:
                nutrition_per_100g = find_nutrition(yolo_detection["food_name"])
                if nutrition_per_100g is not None:
                    return jsonify(build_success_result(
                        yolo_detection["food_name"],
                        yolo_detection["confidence"],
                        weight,
                        nutrition_per_100g=nutrition_per_100g,
                        detected_by=detection_model,
                        recognition_model="yoloworld-detector",
                        message="Used YOLO fruit localization because the generic image model could not match this food.",
                    ))
            result = demo_predict(image_bytes, image_file.filename or "", weight)
            if result:
                result["message"] = "Used heuristic fallback because the generic image model could not match this food."
                return jsonify(result)
            return jsonify({
                "status": "error",
                "message": "Could not recognize a supported fruit or food from this image.",
            }), 422

        if (
            yolo_detection is None
            and heuristic_name
            and heuristic_confidence >= 72
            and (
                confidence < 22
                or food_name != heuristic_name
            )
        ):
            heuristic_nutrition = find_nutrition(heuristic_name)
            if heuristic_nutrition is not None:
                return jsonify(build_success_result(
                    heuristic_name,
                    heuristic_confidence,
                    weight,
                    nutrition_per_100g=heuristic_nutrition,
                    detected_by=heuristic_source,
                    recognition_model="hybrid-heuristic",
                    message="Used fruit-specific visual fallback because the generic image model was uncertain.",
                    demo_mode=True,
                    generic_model_prediction=raw_label,
                    generic_model_confidence=confidence,
                ))

        if yolo_detection is not None and (
            same_food_family(food_name, yolo_detection["food_name"]) or confidence < 35
        ):
            resolved_name = food_name if same_food_family(food_name, yolo_detection["food_name"]) else yolo_detection["food_name"]
            nutrition_per_100g = find_nutrition(resolved_name)
            if nutrition_per_100g is not None:
                return jsonify(build_success_result(
                    resolved_name,
                    max(confidence, yolo_detection["confidence"]),
                    weight,
                    nutrition_per_100g=nutrition_per_100g,
                    detected_by=detection_model,
                    recognition_model="yoloworld+imagenet",
                    message="YOLO localized the fruit before ImageNet classification.",
                ))

        if confidence < 15:
            if yolo_detection is not None:
                nutrition_per_100g = find_nutrition(yolo_detection["food_name"])
                if nutrition_per_100g is not None:
                    return jsonify(build_success_result(
                        yolo_detection["food_name"],
                        max(yolo_detection["confidence"], confidence),
                        weight,
                        nutrition_per_100g=nutrition_per_100g,
                        detected_by=detection_model,
                        recognition_model="yoloworld-detector",
                        message="Used YOLO fruit localization because the generic image model was uncertain.",
                    ))
            fallback_result = demo_predict(image_bytes, image_file.filename or "", weight)
            if fallback_result and fallback_result.get("confidence", 0) >= 70:
                fallback_result["message"] = "Used fruit-specific visual fallback because the generic image model was uncertain."
                fallback_result["recognition_model"] = "hybrid-heuristic"
                fallback_result["generic_model_prediction"] = raw_label
                fallback_result["generic_model_confidence"] = round(confidence, 2)
                return jsonify(fallback_result)

            return jsonify({
                "status":     "low_confidence",
                "food_name":  food_name,
                "confidence": confidence,
                "message":    "The model recognized this fruit with low confidence. Please try a clearer image.",
                "recognition_model": "mobilenetv2-imagenet",
                "raw_label": raw_label,
            })

    nutrition_per_100g = find_nutrition(food_name)
    if nutrition_per_100g is None:
        return jsonify({
            "status":     "not_found",
            "food_name":  food_name,
            "confidence": round(confidence, 2),
            "message":    f"Nutrition data not found for '{food_name}'.",
        })

    return jsonify(build_success_result(
        food_name,
        confidence,
        weight,
        nutrition_per_100g=nutrition_per_100g,
        detected_by=detection_model,
        recognition_model=(
            "yoloworld+custom"
            if image_model_source == "custom" and yolo_detection
            else "custom"
            if image_model_source == "custom"
            else "yoloworld+imagenet"
            if yolo_detection
            else "mobilenetv2-imagenet"
        ),
        message="YOLO localized the fruit before classification." if yolo_detection else None,
    ))


load_resources()


if __name__ == "__main__":
    # threaded=True lets Flask handle concurrent requests
    app.run(debug=True, port=5000, threaded=True)
