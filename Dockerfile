FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py /app/

# Use the PORT environment variable provided by platforms like Render, AWS, or Railway (defaults to 8000)
ENV PORT=8000

# Start the FastAPI app
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
