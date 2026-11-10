# SciLifeLab Serve-compatible Gradio app; Python 3.11, nonroot UID 1000, port 7860.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_TEMP_DIR=/tmp/gradio-tree-crown \
    TREE_APP_OUTPUT_DIR=/tmp/tree-crown-explorer-outputs \
    TREE_DETECTOR_DEVICE=cpu \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser \
    && mkdir -p "$GRADIO_TEMP_DIR" "$TREE_APP_OUTPUT_DIR" \
    && chown -R appuser:appuser "$GRADIO_TEMP_DIR" "$TREE_APP_OUTPUT_DIR"

COPY requirements.txt ./requirements.txt
# CPU-only PyTorch avoids pulling large CUDA dependencies on Serve's CPU profile.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.10.0 torchvision==0.25.0 \
    && python -m pip install --no-cache-dir -r requirements.txt

# Retain the supplied workshop MIT license text in the distributed image.
COPY --chown=1000:1000 LICENSE ./LICENSE
COPY --chown=1000:1000 main.py app_adapter.py tree_crown_detector.py ./
COPY --chown=1000:1000 weights/config.json weights/model.safetensors ./weights/
COPY --chown=1000:1000 sample_images/OSBS_029.png \
    sample_images/SOAP_031.png sample_images/SOAP_061.png \
    sample_images/OSBS_029.tif ./sample_images/

USER 1000:1000
EXPOSE 7860
CMD ["python", "main.py"]
