# ---- Use a lightweight Python base image ----
FROM python:3.11-slim

# ---- Environment settings ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

# ---- Set working directory inside the container ----
WORKDIR /app

# ---- Copy and install Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy all project files into the container ----
COPY . .

# ---- Expose port 8501 (Streamlit default) ----
EXPOSE 8501

# ---- Run the Streamlit app ----
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
