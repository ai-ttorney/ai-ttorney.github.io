# Dockerfile.backend

FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
COPY backend/*.py .
RUN pip install --no-cache-dir -r requirements.txt


COPY backend/ ./backend/


COPY vectorstore/ ./vectorstore/


EXPOSE 8000

# Run your backend server
CMD ["python", "backend/backend_server.py"]