# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (if needed for PDF processing)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main.py file
COPY main.py .

# Create input and output directories for volume mounting
RUN mkdir -p /app/input /app/output

# Run your main.py when container starts
CMD ["python", "main.py"]
