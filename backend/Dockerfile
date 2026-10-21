FROM python:3.11-slim AS build

ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GOLF_DATA_DIR=/data \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app \
    && mkdir -p /app /data/uploads /data/results /data/assets \
    && chown -R app:app /app /data \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --chown=app:app app ./app

USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
