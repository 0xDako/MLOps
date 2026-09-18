FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit \
    streamlit-drawable-canvas \
    httpx \
    opencv-python-headless \
    altair \
    pandas \
    numpy

COPY ui/ ui/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
