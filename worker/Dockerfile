FROM python:3.12-slim

WORKDIR /app

# Install worker deps
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install api deps (shared repository layer)
COPY api/requirements.txt ./api_requirements.txt
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy api source (repository + migrations shared)
COPY api/ /app/api/

# Copy worker source
COPY worker/ /app/

CMD ["python", "main.py"]
