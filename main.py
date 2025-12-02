from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import tensorflow as tf

app = FastAPI()

# ===========================
# LOAD STRONGER MODEL
# ===========================
# EfficientNet B0 Feature Extractor (better than MobileNet)
extractor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
    input_shape=(224, 224, 3),
    trainable=False
)


# -----------------------------------------------------------
# BASE64 → NumPy Tensor
# -----------------------------------------------------------
def decode_base64_image(b64_string: str):
    try:
        if b64_string.startswith("data:image"):
            b64_string = b64_string.split(",")[-1]

        b64_string = b64_string.strip()

        missing_padding = len(b64_string) % 4
        if missing_padding != 0:
            b64_string += "=" * (4 - missing_padding)

        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decode failed")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0

        return np.expand_dims(img, axis=0)

    except Exception as e:
        raise ValueError(f"Base64 decode error: {e}")


# -----------------------------------------------------------
# L2 NORMALIZATION FUNCTION
# Makes embeddings more accurate and stable
# -----------------------------------------------------------
def l2_normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# -----------------------------------------------------------
# API ROUTE
# -----------------------------------------------------------
@app.post("/compare")
@app.post("/compare/")
async def compare_images(payload: dict):
    try:
        if "image1" not in payload or "image2" not in payload:
            return JSONResponse({"error": "Missing image1 or image2"}, 400)

        # Decode images
        img1 = decode_base64_image(payload["image1"])
        img2 = decode_base64_image(payload["image2"])

        # Extract features
        emb1 = extractor(img1).numpy()
        emb2 = extractor(img2).numpy()

        # Normalize vectors
        emb1 = l2_normalize(emb1)
        emb2 = l2_normalize(emb2)

        # Compute cosine similarity
        cos_sim = float(cosine_similarity(emb1, emb2)[0][0])

        # Euclidean distance (lower = similar)
        euclidean_dist = float(np.linalg.norm(emb1 - emb2))

        # FINAL DECISION
        # Must satisfy BOTH strict conditions for a match
        match = cos_sim > 0.80 and euclidean_dist < 0.85

        return {
            "match": match,
            "cosine_similarity": cos_sim,
            "euclidean_distance": euclidean_dist,
        }

    except Exception as e:
        print("🔥 SERVER ERROR:", e)
        return JSONResponse({"error": str(e)}, 500)




























