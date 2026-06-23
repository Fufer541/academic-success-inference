FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# CatBoost and LightGBM depend on the OpenMP runtime. The slim Python image does
# not include it by default, so install the smallest Debian package that provides
# `libgomp.so.1`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the API code into the image. The trained model bundle is mounted at
# runtime by Docker Compose so large artifacts do not have to be baked into the
# backend image.
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]