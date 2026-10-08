"""VoxCPM2 Studio: inferencia de voz y generación de datasets Piper."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import gradio as gr
import numpy as np
import soundfile as sf

OUTPUTS = Path(os.getenv("VOXCPM_OUTPUTS", "/kaggle/working/voxcpm2_outputs"))
MAX_ROWS = 5000
MAX_CSV_SIZE = 2 * 1024 * 1024
MAX_TEXT_LEN = 400
MAX_AUDIO_SIZE = 30 * 1024 * 1024

_model = None
_model_mutex = threading.RLock()


def parse_texts(csv_path: str) -> list[str]:
    """Accept UTF-8 CSV text, texto, frase, id|text, or a single text column."""
    if not csv_path:
        raise ValueError("Carga un archivo CSV.")
    source = Path(csv_path)
    if source.stat().st_size > MAX_CSV_SIZE:
        raise ValueError("El CSV supera el límite de 2 MB.")
    content = source.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(content[:8192], delimiters=",;|\t")
    except csv.Error:
        dialect = csv.excel
    # A one-column CSV commonly contains unquoted commas inside sentences.
    # In that case preserve each complete line rather than splitting punctuation.
    raw_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if raw_lines and raw_lines[0].strip().lower() in {"texto", "text", "frase", "sentence"}:
        return validate_texts(raw_lines[1:]) if not any(line.startswith('"') for line in raw_lines[1:]) else validate_texts([row[0] for row in csv.reader(io.StringIO(content), dialect)][1:])
    rows = list(csv.reader(io.StringIO(content), dialect))
    if not rows:
        raise ValueError("El CSV está vacío.")
    header = [h.strip().lower() for h in rows[0]]
    text_index = next(
        (i for i, x in enumerate(header) if x in ("text", "texto", "frase", "sentence", "transcript", "transcripción")),
        None,
    )
    if text_index is not None:
        data = rows[1:]
        index = text_index
    else:
        data = rows
        index = 1 if len(rows[0]) > 1 else 0
    if len(data) > MAX_ROWS:
        raise ValueError(f"El máximo por lote es {MAX_ROWS} filas.")
    texts = []
    for position, row in enumerate(data, 1):
        if not row or len(row) <= index or not row[index].strip():
            continue
        value = row[index].strip()
        if len(value) > MAX_TEXT_LEN:
            raise ValueError(f"Fila {position}: excede {MAX_TEXT_LEN} caracteres.")
        if "|" in value or "\n" in value or "\r" in value:
            raise ValueError(f"Fila {position}: contiene separadores incompatibles con metadata de Piper.")
        texts.append(value)
    return validate_texts(texts)


def validate_texts(texts):
    if len(texts) > MAX_ROWS:
        raise ValueError(f"El máximo por lote es {MAX_ROWS} filas.")
    for index, value in enumerate(texts, 1):
        if len(value) > MAX_TEXT_LEN or "|" in value or "\\n" in value or "\\r" in value:
            raise ValueError(f"Fila {index}: texto inválido o con separadores incompatibles con metadata de Piper.")
    if not texts:
        raise ValueError("No se encontraron frases en el CSV.")
    return texts


def check_reference(reference_path: str | None, transcript: str, mode: str) -> None:
    if mode == "Sin referencia":
        return
    if not reference_path:
        raise ValueError("Carga un audio de referencia para usar clonación.")
    source = Path(reference_path)
    if not source.is_file() or source.stat().st_size > MAX_AUDIO_SIZE:
        raise ValueError("El archivo de referencia no existe o supera 30 MB.")
    if mode == "Clonación precisa" and not transcript.strip():
        raise ValueError("La clonación precisa necesita la transcripción EXACTA del audio de referencia.")


def load_model():
    """Loaded lazily; only one model is kept in memory and one generation runs at a time."""
    global _model
    if _model is None:
        import torch
        if not torch.cuda.is_available() and os.environ.get("VOXCPM_ALLOW_CPU") != "1":
            raise RuntimeError(
                "CUDA no está disponible en ESTE proceso. Activa GPU en Kaggle o ejecuta "
                "VOXCPM_ALLOW_CPU=1 para pruebas en CPU (muy lentas)."
            )
        from voxcpm import VoxCPM
        _model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)
    return _model


def synthesize(text: str, reference: str | None, transcript: str,
               mode: str, cfg: float, steps: int, seed: int) -> tuple[np.ndarray, int]:
    check_reference(reference, transcript, mode)
    with _model_mutex:
        voice = load_model()
        import random
        import torch
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        torch.manual_seed(int(seed))
        kwargs = dict(
            text=text, cfg_value=float(cfg), inference_timesteps=int(steps),
        )
        if mode != "Sin referencia":
            kwargs["reference_wav_path"] = reference
        if mode == "Clonación precisa":
            kwargs["prompt_wav_path"] = reference
            kwargs["prompt_text"] = transcript.strip()
        samples = voice.generate(**kwargs)
        sample_rate = int(voice.tts_model.sample_rate)
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(samples) or not np.isfinite(samples).all():
        raise RuntimeError("El modelo devolvió audio vacío o no válido.")
    return samples, sample_rate


def save_wav(samples: np.ndarray, sr: int, dest: Path, output_sr: int = 22050) -> None:
    """Resample for Piper without changing playback speed."""
    from scipy.signal import resample_poly
    from math import gcd
    if sr != output_sr:
        factor = gcd(sr, output_sr)
        samples = resample_poly(samples, output_sr // factor, sr // factor)
    samples = np.clip(samples, -1, 1)
    sf.write(dest, samples, output_sr, subtype="PCM_16")


def synthesize_one(text, audio, transcript, mode, cfg, steps, seed):
    text = (text or "").strip()
    if not text or len(text) > MAX_TEXT_LEN:
        raise gr.Error(f"Escribe entre 1 y {MAX_TEXT_LEN} caracteres.")
    try:
        run_dir = OUTPUTS / ("prueba_" + uuid.uuid4().hex[:12])
        run_dir.mkdir(parents=True, exist_ok=False)
        samples, sr = synthesize(text, audio, transcript or "", mode, cfg, steps, seed)
        path = run_dir / "inferencia.wav"
        sf.write(path, np.clip(samples, -1, 1), sr, subtype="PCM_16")
        return str(path), f"Audio creado: {sr} Hz, {len(samples) / sr:.1f} segundos. Se conservó la frecuencia original."
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def generate_dataset(csv_path, audio, transcript, mode, cfg, steps, seed,
                     sample_rate, progress=gr.Progress()):
    try:
        texts = parse_texts(csv_path)
        check_reference(audio, transcript or "", mode)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    job = OUTPUTS / ("dataset_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    wavs_dir = job / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=False)
    lines = []
    failures = []
    preview = None
    for index, text in enumerate(texts, 1):
        progress((index - 1, len(texts)), desc=f"Generando {index}/{len(texts)}")
        # Different deterministic seed per sentence; reproducible across reruns.
        row_seed = int(seed) + index - 1
        wav_id = f"{index:06d}"
        dest = wavs_dir / f"{wav_id}.wav"
        try:
            samples, sr = synthesize(text, audio, transcript or "", mode, cfg, steps, row_seed)
            save_wav(samples, sr, dest, output_sr=int(sample_rate))
            lines.append(f"{wav_id}|{text}")
            if preview is None:
                preview = str(dest)
        except Exception as exc:
            failures.append(dict(row=index, message=str(exc)[:400]))
            if dest.exists():
                dest.unlink()
            # Stop on infrastructure/model failures instead of producing thousands of errors.
            if index == 1:
                break

    (job / "metadata.csv").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (job / "report.json").write_text(
        json.dumps(dict(generated=len(lines), requested=len(texts), errors=failures,
                        reference_included=False, model="openbmb/VoxCPM2",
                        mode=mode, sample_rate=sample_rate), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not lines:
        raise gr.Error("No se generaron WAV válidos. " + (failures[0]["message"] if failures else "Error desconocido."))

    archive = job / "piper_dataset.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        zf.write(job / "metadata.csv", "metadata.csv")
        zf.write(job / "report.json", "report.json")
        for wav in sorted(wavs_dir.glob("*.wav")):
            zf.write(wav, f"wavs/{wav.name}")
    progress((len(texts), len(texts)), desc="Completado")
    report = (
        f"Generados: {len(lines)}/{len(texts)} WAV · "
        f"Errores: {len(failures)} · {sample_rate} Hz.\n"
        f"Carpeta: {job}\n"
        "El ZIP contiene wavs/, metadata.csv y report.json. No incluye la voz de referencia."
    )
    return str(archive), preview, report


def inspect_csv(path):
    try:
        texts = parse_texts(path)
        first = "\n".join(f"{i+1}. {s}" for i, s in enumerate(texts[:8]))
        return f"{len(texts)} frases válidas.\n\n{first}"
    except Exception as exc:
        return f"CSV no válido: {exc}"


CSS = """
.gradio-container {max-width: 1120px !important}
#hero {padding:20px 24px;border-radius:14px;background:linear-gradient(120deg,#162943,#0a5e67);color:#fff}
#hero h1,#hero p {color:white !important}
"""


def build_ui():
    with gr.Blocks(title="VoxCPM2 | Dataset Studio") as ui:
        gr.HTML(
            '<section id="hero"><h1>VoxCPM2 · Dataset Studio</h1>'
            '<p>Clonación de voz, inferencia y datasets WAV/CSV para Piper-Neo.</p></section>'
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Voz y parámetros")
                mode = gr.Radio(
                    ["Clonación precisa", "Clonación por referencia", "Sin referencia"],
                    value="Clonación precisa", label="Método de voz",
                )
                reference = gr.Audio(
                    label="Tu voz de referencia", sources=["upload", "microphone"], type="filepath",
                )
                transcript = gr.Textbox(
                    label="Transcripción exacta del audio de referencia",
                    placeholder="Escribe exactamente lo que dices en el audio...",
                    lines=3,
                )
                whisper_button = gr.Button("Transcribir referencia automáticamente (Whisper large-v3)")
                whisper_status = gr.Textbox(label="Estado de Whisper", interactive=False)
                from whisper_transcribe import transcribe_on_audio_change, transcribe_reference
                whisper_button.click(
                    transcribe_reference, inputs=[reference], outputs=[transcript, whisper_status],
                    concurrency_id="whisper_gpu", concurrency_limit=1,
                )
                reference.change(
                    transcribe_on_audio_change, inputs=[reference], outputs=[transcript, whisper_status],
                    concurrency_id="whisper_gpu", concurrency_limit=1,
                )
                cfg = gr.Slider(1.0, 4.0, value=2.0, step=0.1, label="CFG / adherencia")
                steps = gr.Slider(5, 30, value=10, step=1, label="Pasos de inferencia")
                seed = gr.Number(value=42, precision=0, label="Semilla")
                gr.Markdown(
                    "Para **Clonación precisa** sube un WAV con voz limpia y escribe lo que "
                    "se oye, palabra por palabra. Usa solo voces propias o autorizadas."
                )
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Probar voz"):
                        text = gr.Textbox(
                            label="Texto para sintetizar", lines=5,
                            value="Hola, esta es una prueba de mi nueva voz en español latinoamericano.",
                        )
                        try_btn = gr.Button("Generar prueba", variant="primary")
                        out_audio = gr.Audio(label="Resultado de inferencia", type="filepath")
                        out_status = gr.Textbox(label="Estado", interactive=False)
                        try_btn.click(
                            synthesize_one,
                            inputs=[text, reference, transcript, mode, cfg, steps, seed],
                            outputs=[out_audio, out_status],
                            concurrency_id="voxcpm_gpu", concurrency_limit=1,
                        )
                    with gr.Tab("Generar dataset Piper"):
                        gr.Markdown(
                            "Acepta CSV con columna **texto/text/frase** o filas como "
                            "`id|texto`. Exporta `wavs/000001.wav` y `metadata.csv`."
                        )
                        csv_file = gr.File(label="Archivo CSV", file_types=[".csv"], type="filepath")
                        preview = gr.Textbox(label="Vista previa del CSV", lines=7, interactive=False)
                        csv_file.change(inspect_csv, inputs=csv_file, outputs=preview)
                        rate = gr.Dropdown([22050, 24000], value=22050, label="Frecuencia WAV para Piper (Hz)")
                        batch_btn = gr.Button("Generar todos los WAV", variant="primary")
                        zip_out = gr.File(label="Descargar ZIP compatible con Piper")
                        audio_preview = gr.Audio(label="Primer audio generado", type="filepath")
                        report = gr.Textbox(label="Informe del lote", lines=4, interactive=False)
                        batch_btn.click(
                            generate_dataset,
                            inputs=[csv_file, reference, transcript, mode, cfg, steps, seed, rate],
                            outputs=[zip_out, audio_preview, report],
                            concurrency_id="voxcpm_gpu", concurrency_limit=1,
                        )
        gr.Markdown(
            "**Privacidad:** los audios de referencia no se añaden al ZIP. "
            "Si habilitas Share, el enlace puede estar disponible públicamente: "
            "establece STUDIO_USER y STUDIO_PASSWORD antes de compartirlo. "
            "Una tarea interrumpida deja sus WAV ya generados en la carpeta de salidas de Kaggle."
        )
    return ui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Publicar enlace temporal de Gradio")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    user = os.getenv("STUDIO_USER")
    password = os.getenv("STUDIO_PASSWORD")
    if bool(user) != bool(password):
        raise SystemExit("Establece STUDIO_USER y STUDIO_PASSWORD conjuntamente.")
    if args.share and not (user and password):
        print("AVISO: share público sin autenticación. No subas voces sensibles.", flush=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    auth = (user, password) if user and password else None
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share,
        auth=auth, show_error=True, prevent_thread_lock=False,
        css=CSS, theme=gr.themes.Soft(),
        # Gradio 6 requires explicitly allowing files returned outside its working dir.
        # Limit exposure to the application output directory, never /kaggle/working.
        allowed_paths=[str(OUTPUTS.resolve())],
    )


if __name__ == "__main__":
    main()
