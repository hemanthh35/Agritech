# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Expose port
EXPOSE 5000

# Start with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
