# Minimal, non-root image for agent/sandbox.py's run_python tool.
# Isolation (--network none, resource limits, capability drop) is applied by
# the `docker run` invocation in sandbox.py, not by this image.
FROM python:3.11-slim

# numpy is already a transitive dependency of pandas/scipy - listed
# explicitly (unpinned) so pip resolves one version satisfying all three
# instead of it being an implicit, easy-to-forget side effect.
# python-docx reads .docx tables directly (pandas has no reader for the
# format) - lets generated code compute over a DOCX table, not just search/
# read its text.
RUN pip install --no-cache-dir pandas==2.2.* openpyxl==3.1.* scipy==1.14.* numpy python-docx==1.1.*

RUN useradd --uid 1000 --create-home sandbox
USER sandbox

WORKDIR /sandbox
# Mount point for the writable scratch tmpfs sandbox.py attaches at run time;
# the rest of the filesystem stays read-only.
RUN mkdir -p /sandbox/work
