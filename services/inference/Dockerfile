FROM python:3.12-slim

WORKDIR /app

COPY README.md /app/README.md

EXPOSE 8003

CMD ["python", "-m", "http.server", "8003", "--bind", "0.0.0.0"]
