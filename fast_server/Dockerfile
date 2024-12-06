FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY main.py .
COPY utils.py .
COPY model model

EXPOSE 8080

# Run FastAPI app
CMD ["python", "main.py"]
