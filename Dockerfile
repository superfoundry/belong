# The hosted demo (README, "Run it online"). Pure standard library: nothing to install.
# Keys and the demo password come from the host's secret store at run time, never from this image.
FROM python:3.13-slim
WORKDIR /app
COPY belong ./belong
COPY README.md SPEC.md TRUST.md ./
# The inspector's batch run, made once at build time.
RUN python3 -m belong run
ENV PYTHONUNBUFFERED=1 BELONG_HOST=0.0.0.0 BELONG_REQUIRE_PASSWORD=1
# Cloud Run says which port to use in $PORT.
CMD ["python3", "-m", "belong", "serve"]
