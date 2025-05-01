# Build stage
FROM python:3.12.4-slim AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /build

# Cài đặt các gói cần thiết cho việc build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies vào virtual environment
COPY requirements.txt .
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.12.4-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Cài đặt thư viện runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment từ build stage
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy code
COPY . /app

# Expose port
EXPOSE 8000

# Chạy ứng dụng
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}