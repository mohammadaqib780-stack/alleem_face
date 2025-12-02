FROM python:3.10

# Prevent TensorFlow from trying to use GPU inside Docker
ENV TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

COPY requirements.txt .

# Install system dependencies for OpenCV, TensorFlow & SciPy
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
