FROM python:3.10-slim

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy application code and files
COPY . .

# 3. Expose port 8000
EXPOSE 8000

# 4. Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
