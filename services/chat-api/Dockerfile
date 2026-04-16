FROM python:3.12-slim

WORKDIR /app

COPY README.md /app/README.md

EXPOSE 8002

CMD ["python", "-m", "http.server", "8002", "--bind", "0.0.0.0"]
