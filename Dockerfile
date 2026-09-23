# Imagen del algoritmo C2D «fl-local-train».
# El data owner aprueba esta imagen por su DIGEST (sha256), no por el tag:
# cualquier cambio en train.py o fl_model.py produce un digest nuevo.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

# torch CPU (índice oficial de PyTorch, sin dependencias CUDA → imagen ~ 800 MB en vez de ~ 5 GB)
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install --no-cache-dir numpy==2.1.3 safetensors==0.4.5

WORKDIR /app
COPY fl_model.py train.py /app/

# ocean-node ejecuta el contenedor como 1000:1000, sin red y con CapDrop=ALL
USER 1000:1000
ENTRYPOINT ["python", "/app/train.py"]
