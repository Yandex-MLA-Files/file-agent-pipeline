# Minimal, non-root image for agent/sandbox.py's run_python tool.
# Isolation (--network none, resource limits, capability drop) is applied by
# the `docker run` invocation in sandbox.py, not by this image.
FROM python:3.11-slim

RUN pip install --no-cache-dir pandas==2.2.* openpyxl==3.1.*

RUN useradd --uid 1000 --create-home sandbox
USER sandbox

WORKDIR /sandbox
