FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python packages (CPU-only PyTorch)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "proxserver.py"]