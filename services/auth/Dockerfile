FROM python:3.12-slim

WORKDIR /app

COPY README.md /app/README.md

EXPOSE 8001

CMD ["python", "-m", "http.server", "8001", "--bind", "0.0.0.0"]
