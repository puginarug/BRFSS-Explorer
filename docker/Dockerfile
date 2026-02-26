# ============================================================
# ScreenMind Dockerfile — skeleton
# Every line will be explained in detail in Milestone 8.
# ============================================================

# Base image: official Python 3.11 slim variant (smaller than full)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list first (lets Docker cache this layer
# so reinstalls are skipped if requirements.txt hasn't changed)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Expose the port FastAPI will listen on
EXPOSE 8000

# Command to start the server
# --host 0.0.0.0 makes it reachable from outside the container
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
