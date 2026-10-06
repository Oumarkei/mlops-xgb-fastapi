# 1. Base Image
FROM python:3.13-slim

#2. Working Directory
WORKDIR /app

# 3. Dependencies layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Source code layer
COPY app.py best_pipeline.pkl ./

# 5. Port Exposure
EXPOSE 8000

# 6. Execution command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0","--port","8000"]

