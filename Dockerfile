# syntax=docker/dockerfile:1
# Follows config/template.Dockerfile of esl-epfl/szcore. CPU only; no network access is needed at run time.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base
LABEL org.opencontainers.image.title="RelSz" \
      org.opencontainers.image.description="Causal patient-relative seizure detector for scalp EEG (SzCORE format)" \
      org.opencontainers.image.source="https://github.com/guhanvenkat10/relsz" \
      org.opencontainers.image.licenses="CC-BY-NC-4.0" \
      org.opencontainers.image.version="0.2.0"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
ARG UID=10001
RUN adduser --disabled-password --gecos "" --home "/nonexistent" --shell "/sbin/nologin" --no-create-home --uid "${UID}" appuser
COPY LICENSE pyproject.toml /app/
COPY relsz /app/relsz
RUN python -m pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu && python -m pip install --no-cache-dir /app
USER appuser
VOLUME ["/data"]
VOLUME ["/output"]
ENV INPUT=""
ENV OUTPUT=""
CMD python3 -m relsz "/data/${INPUT}" "/output/${OUTPUT}"
