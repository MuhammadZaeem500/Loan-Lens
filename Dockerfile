FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything in the current directory into /app
COPY . .

# Use python -m uvicorn to ensure it runs within the correct Python path context
CMD ["python", "-m", "uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]