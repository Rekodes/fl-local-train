# fl-local-train

Algoritmo Compute-to-Data de ocean-node para el entrenamiento local de un sitio en aprendizaje
federado (FedAvg/FedProx). Es el código de la imagen pública
[`rekodes/fl-local-train`](https://hub.docker.com/r/rekodes/fl-local-train).

Este repositorio existe para que quien aprueba el algoritmo pueda leer exactamente lo que corre:
el `code_url` del algoritmo publicado apunta a `train.py` en un commit concreto, y su contenido no
cambia. La copia de trabajo vive en `Rekodes/ocean-fl` (`algorithm/`).

| Versión | Imagen (linux/amd64) |
|---|---|
| 0.4.0 | `rekodes/fl-local-train@sha256:952454acc822305cb2eeee8791244d25c5b67ccce6600366beaa7a3e39316a8f` |

## Contrato

**Entradas** (solo lectura):
- `/data/inputs/algoCustomData.json`: hiperparámetros de la ronda y el modelo global en `model`
  (safetensors en base64), verificado contra `model_ref` (`sha256:<hex>`) si llega.
- `/data/inputs/<datos>.csv`: datos del sitio, un único fichero, con la etiqueta en la última
  columna.
- Alternativa del laboratorio: el modelo en `/data/persistentStorage/<bucket>/<fichero>`.

**Salidas** (esquema cerrado, nada más sale del sitio):
- `delta_rNNNN.safetensors`: Δw = w_local − w_t, float32, con las mismas claves que el modelo.
- `metrics_rNNNN.json`: `model_ref` del modelo recibido, `n_train`, `n_val`, pérdida y exactitud
  de validación antes y después, norma del delta y tiempos. Revela `n_train`, que FedAvg necesita.

**Límites** fijados en la imagen, aunque se pidan otros: épocas 1–50, `lr` 1e-5–1, lote 16–4096,
fracción de validación 0,05–0,5, μ de FedProx 0–10. Las claves que no están en la lista se ignoran.

El contenedor corre sin red, sin capabilities y como UID 1000. Cualquier cambio en `train.py` o
`fl_model.py` produce una imagen con otro digest y exige una nueva aprobación de cada dueño del dato.
