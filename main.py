from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import tensorflow as tf

app = FastAPI()

# Load model once
extractor = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    input_shape=(224,224,3),
    trainable=False
)

def decode_base64(b64):
    try:
        b64 = b64.replace("\n", "").replace(" ", "")  # clean
        img_data = base64.b64decode(b64 + "===")
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = img / 255.0

        return np.expand_dims(img, 0)
    except Exception as e:
        raise ValueError(f"Base64 error: {str(e)}")

@app.post("/compare")
async def compare(payload: dict):
    try:
        img1 = decode_base64(payload["image1"])
        img2 = decode_base64(payload["image2"])

        emb1 = extractor(img1).numpy()
        emb2 = extractor(img2).numpy()

        similarity = cosine_similarity(emb1, emb2)[0][0]
        match = similarity > 0.60

        return {"match": match, "similarity": float(similarity)}

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)
