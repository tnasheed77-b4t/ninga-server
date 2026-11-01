FROM python:3.12-slim

# Copy Deno binary into Linux PATH for yt-dlp signature solving
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

# Install system packages cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Always upgrade yt-dlp to the latest nightly build (equivalent to yt-dlp -U --update-to nightly)
RUN pip install --no-cache-dir -U --pre "yt-dlp[default,curl-cffi]"

# Run uvicorn using exec array format for proper signal propagation
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]