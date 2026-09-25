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
| 0.4.1 | `rekodes/fl-local-train@sha256:384de755428a1c200a06f1b4d3fba42ff81cbffeb30de6a39235683a4856b2c5` |

## Contrato

**Entradas** (solo lectura):
- `/data/inputs/algoCustomData.json`: hiperparámetros de la ronda y el modelo global en `model`
  (safetensors en base64), verificado contra `model_ref` (`sha256:<hex>`) si llega.
- `/data/inputs/<datos>.csv`: datos del sitio, un único fichero, en formato `csv-num20-label10`:
  cabecera, 20 columnas numéricas y la etiqueta 0..9 al final. Desde la 0.4.1, otro formato sale
  con código 1 y una sola línea en el log que dice qué falla.
- Alternativa del laboratorio: el modelo en `/data/persistentStorage/<bucket>/<fichero>`.

**Salidas** (esquema cerrado, nada más sale del sitio):
- `delta_rNNNN.safetensors`: Δw = w_local − w_t, float32, con las mismas claves que el modelo.
- `metrics_rNNNN.json`: `model_ref` del modelo recibido, `n_train`, `n_val`, pérdida y exactitud
  de validación antes y después, norma del delta y tiempos. Revela `n_train`, que FedAvg necesita.

**Límites** fijados en la imagen, aunque se pidan otros: épocas 1–50, `lr` 1e-5–1, lote 16–4096,
fracción de validación 0,05–0,5, μ de FedProx 0–10. Las claves que no están en la lista se ignoran.

El contenedor corre sin red, sin capabilities y como UID 1000. Cualquier cambio en `train.py` o
`fl_model.py` produce una imagen con otro digest y exige una nueva aprobación de cada dueño del dato.

## Datos de prueba: SYNTH-10

`data/synth10/alpha_1/`: el dataset sintético SYNTH-10 (clasificación tabular, 10 clases, 20
variables) repartido en tres sitios con una partición de Dirichlet de α = 1 (semilla 11), con
10 618, 10 053 y 9 329 filas. Lo generan `data/generate_dataset.py` y `data/partition_dirichlet.py`
de `Rekodes/ocean-fl`; `stats.json` trae el recuento por clase de cada sitio.

Son **datos sintéticos y públicos**, para probar el aprendizaje federado de punta a punta: cada
dataset publicado apunta a su fichero en un commit concreto. Nunca se publican así datos reales.
