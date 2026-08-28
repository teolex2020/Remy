FROM python:3.12-alpine@sha256:d09d15e60962ca365d1cd544a48773bac9d33f2fb1b00f2aa0deec78ade7dc31

LABEL org.opencontainers.image.title="Remy Agent Lab Runtime"
LABEL org.opencontainers.image.description="Dependency-free Python boundary for local Agent Lab execution"
LABEL org.opencontainers.image.vendor="Remy"

ENV PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace
ENTRYPOINT []
CMD ["python", "-I", "-S", "-B"]
