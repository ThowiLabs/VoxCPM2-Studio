"""Transcripción local con Whisper large-v3; se carga únicamente al solicitarla."""
from pathlib import Path
import gc
import os
import threading

_LOCK = threading.Lock()


def transcribe_on_audio_change(audio_path: str | None) -> tuple[str, str]:
    """Clear text without starting Whisper when the user removes the audio."""
    if not audio_path:
        return "", ""
    return transcribe_reference(audio_path)


def transcribe_reference(audio_path: str | None) -> tuple[str, str]:
    if not audio_path or not Path(audio_path).is_file():
        raise ValueError("Sube primero un audio de referencia.")
    import torch
    from faster_whisper import WhisperModel

    model_name = os.getenv("WHISPER_MODEL", "large-v3")
    # A Kaggle T4 is sufficient with float16. CUDA 1 avoids interfering
    # with VoxCPM on CUDA 0 when both are available.
    if torch.cuda.is_available():
        device = "cuda"
        device_index = 1 if torch.cuda.device_count() > 1 else 0
        compute_type = "float16"
    else:
        device = "cpu"
        device_index = 0
        compute_type = "int8"

    with _LOCK:
        model = WhisperModel(
            model_name, device=device, device_index=device_index,
            compute_type=compute_type,
        )
        try:
            segments, info = model.transcribe(
                audio_path,
                language="es",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            result = " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not result:
        raise ValueError("Whisper no detectó voz reconocible en el archivo.")
    return result, f"Whisper {model_name} · idioma español · {device}:{device_index}. Revisa el texto antes de clonar."
