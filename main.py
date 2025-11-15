from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import tensorflow as tf

app = FastAPI()

# Load MobileNetV2 once
extractor = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    input_shape=(224,224,3),
    trainable=False
)

def decode_base64_image(b64_string: str):
    try:
        # Remove metadata prefix "data:image/jpeg;base64,"
        if b64_string.startswith("data:image"):
            b64_string = b64_string.split(",")[-1]

        b64_string = b64_string.strip()

        # Fix padding issues automatically
        missing_padding = len(b64_string) % 4
        if missing_padding != 0:
            b64_string += "=" * (4 - missing_padding)

        # Decode base64 → image bytes
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decode failed")

        # Convert and preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0

        return np.expand_dims(img, axis=0)

    except Exception as e:
        raise ValueError(f"Base64 decode error: {e}")

@app.post("/compare")
@app.post("/compare/")
async def compare_images(payload: dict):
    try:
        if "image1" not in payload or "image2" not in payload:
            return JSONResponse({"error": "Missing image1 or image2"}, 400)

        # Decode base64 images
        img1 = decode_base64_image(payload["image1"])
        img2 = decode_base64_image(payload["image2"])

        # Extract embeddings
        emb1 = extractor(img1).numpy()
        emb2 = extractor(img2).numpy()

        # Cosine similarity
        similarity = float(cosine_similarity(emb1, emb2)[0][0])
        match = similarity > 0.60

        return {"match": match, "similarity": similarity}

    except Exception as e:
        print("🔥 SERVER ERROR:", e)
        return JSONResponse({"error": str(e)}, 500)
