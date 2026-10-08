# VoxCPM2 Studio

Repositorio previsto: https://github.com/ThowiLabs/VoxCPM2-Studio

VoxCPM2 + Gradio 6 para clonar una voz autorizada, probar frases y generar WAV + metadata.csv compatibles con Piper-Neo.

## Novedades

- Whisper **large-v3** mediante faster-whisper transcribe automáticamente el audio de referencia al subirlo. Se fija **PyAV 18.1.0** porque PyAV 19 elimina el parámetro `metadata_errors` utilizado por faster-whisper 1.2.1; si actualizas las dependencias, conserva esa compatibilidad. También puedes repetir la transcripción con el botón y editar el resultado para corregir palabras.
- Usa preferentemente la GPU 1 para Whisper y deja la GPU 0 para VoxCPM. Si solo existe una GPU, libera el modelo Whisper al terminar.
- CSV de frases con comas: columna texto, text o frase; también formato ID|texto.
- Exporta ZIP con metadata.csv, report.json y wavs/ PCM16 en 22050 o 24000 Hz.
- Referencias vocales no se incluyen en el ZIP.

## Kaggle Save Version > Save & Run All

Activa GPU e Internet y abre KAGGLE_VOXCPM2_STUDIO.ipynb. El notebook busca un ZIP adjunto llamado VoxCPM2-Studio-GIT.zip o clona el repositorio GitHub cuando ya esté publicado. Instala las dependencias y comprueba CUDA.

**La última celda ejecuta Gradio mediante subprocess.run(), en primer plano.** Así la ejecución versionada continúa activa mientras Gradio funcione. La URL pública de gradio.live aparecerá en el log de esa celda. Es temporal: depende de la cuota, duración de la ejecución y túnel de Gradio.

El acceso se inicia sin autenticación a menos que configures STUDIO_USER y STUDIO_PASSWORD como secretos de Kaggle. Si Share no está protegido, cualquiera con la URL podrá entrar.

## Instalación manual

```bash
bash install_kaggle.sh
.venv/bin/python -u studio.py --share
```

La primera inferencia descarga los pesos de VoxCPM2; la primera transcripción descarga Whisper large-v3. La GPU es necesaria para un rendimiento práctico.

## Pruebas

```bash
.venv/bin/python -m pytest -q test_studio.py
.venv/bin/python -m py_compile studio.py whisper_transcribe.py
```

## GitHub

La entrega contiene un solo commit inicial. Autor confirmado en commits públicos de Piper-Neo: ThowiLabs <291061271+ThowiLabs@users.noreply.github.com>.

Para publicar desde una extracción del ZIP Git:

```bash
git remote add origin https://github.com/ThowiLabs/VoxCPM2-Studio.git
git push -u origin main
```

Solo utilizar voces propias o autorizadas. No publicar grabaciones, modelos descargados ni tokens en el repositorio.
