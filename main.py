# from fastapi import FastAPI
# from fastapi.responses import JSONResponse
# import numpy as np
# import cv2
# import base64
# from sklearn.metrics.pairwise import cosine_similarity
# import tensorflow_hub as hub
# import tensorflow as tf

# app = FastAPI()

# # ===========================
# # LOAD EFFICIENTNET B0
# # ===========================
# extractor = hub.KerasLayer(
#     "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
#     input_shape=(224, 224, 3),
#     trainable=False
# )

# # EfficientNet preprocessing values
# MEAN = np.array([0.485, 0.456, 0.406])
# STD = np.array([0.229, 0.224, 0.225])


# # -----------------------------------------------------------  
# # BASE64 → TENSOR  
# # -----------------------------------------------------------
# def decode_image(b64_string: str):
#     try:
#         if b64_string.startswith("data:image"):
#             b64_string = b64_string.split(",")[-1]

#         b64_string = b64_string.strip()
#         missing_padding = len(b64_string) % 4
#         if missing_padding != 0:
#             b64_string += "=" * (4 - missing_padding)

#         img_bytes = base64.b64decode(b64_string)
#         img_array = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#         if img is None:
#             raise ValueError("Failed to decode")

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (224, 224))

#         img = img.astype("float32") / 255.0
#         img = (img - MEAN) / STD   # EfficientNet normalization

#         return np.expand_dims(img, axis=0)

#     except Exception as e:
#         raise ValueError(f"Base64 decode error: {e}")


# # -----------------------------------------------------------  
# # L2 NORMALIZATION  
# # -----------------------------------------------------------
# def l2_norm(v):
#     return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)


# # -----------------------------------------------------------  
# # FACE / PET MATCHING ROUTE  
# # -----------------------------------------------------------
# @app.post("/compare")
# async def compare_images(payload: dict):
#     try:
#         if "image1" not in payload or "image2" not in payload:
#             return JSONResponse({"error": "Missing image1/image2"}, 400)

#         img1 = decode_image(payload["image1"])
#         img2 = decode_image(payload["image2"])

#         emb1 = extractor(img1).numpy()
#         emb2 = extractor(img2).numpy()

#         emb1 = l2_norm(emb1)
#         emb2 = l2_norm(emb2)

#         # Similarity metrics
#         cos_sim = float(cosine_similarity(emb1, emb2)[0][0])
#         dist = float(np.linalg.norm(emb1 - emb2))

#         # Improved thresholds (tested for animals)
#         if cos_sim > 0.80 and dist < 0.75:
#          match = True
#         else:
#          match = False

#         return {
#             "match": match,
#             "cosine_similarity": cos_sim,
#             "distance": dist,
#         }

#     except Exception as e:
#         return JSONResponse({"error": str(e)}, 500)




from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch

app = FastAPI()

# ===========================
# LOAD CLIP MODEL (BEST OPTION)
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# -----------------------------------------------------------
# BASE64 → IMAGE ARRAY
# -----------------------------------------------------------
def decode_image(b64_string: str):
    try:
        if b64_string.startswith("data:image"):
            b64_string = b64_string.split(",")[-1]

        missing_padding = len(b64_string) % 4
        if missing_padding != 0:
            b64_string += "=" * (4 - missing_padding)

        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decode failed")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    except Exception as e:
        raise ValueError(f"Base64 decode error: {e}")


# -----------------------------------------------------------
# EXTRACT CLIP EMBEDDINGS
# -----------------------------------------------------------
def embed_image(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # L2 normalize
    return emb.cpu().numpy()


# -----------------------------------------------------------
# MAIN MATCHING ROUTE
# -----------------------------------------------------------
@app.post("/compare")
async def compare_images(payload: dict):
    try:
        if "image1" not in payload or "image2" not in payload:
            return JSONResponse({"error": "Missing image1/image2"}, 400)

        img1 = decode_image(payload["image1"])
        img2 = decode_image(payload["image2"])

        emb1 = embed_image(img1)
        emb2 = embed_image(img2)

        cos_sim = float(cosine_similarity(emb1, emb2)[0][0])
        dist = float(np.linalg.norm(emb1 - emb2))

        # MUCH BETTER THRESHOLDS USING CLIP
        match = cos_sim > 0.70 and dist < 1.20

        return {
            "match": match,
            "cosine_similarity": cos_sim,
            "distance": dist,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)
