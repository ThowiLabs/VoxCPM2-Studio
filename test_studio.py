from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import csv
import io
import zipfile

import numpy as np
import pytest
import studio


def test_header_csv(tmp_path):
    source = tmp_path / "metadata.csv"
    source.write_text("texto\nHola mundo\n¿Cómo estás?\n", encoding="utf-8")
    assert studio.parse_texts(str(source)) == ["Hola mundo", "¿Cómo estás?"]


def test_csv_one_column_with_commas(tmp_path):
    source = tmp_path / "frases.csv"
    source.write_text("texto\nHola, ¿cómo estás?\nSí, claro.\n", encoding="utf-8")
    assert studio.parse_texts(str(source)) == ["Hola, ¿cómo estás?", "Sí, claro."]


def test_piper_pipe_csv(tmp_path):
    source = tmp_path / "metadata.csv"
    source.write_text("000001|Hola mundo\n000002|Prueba de voz\n", encoding="utf-8")
    assert studio.parse_texts(str(source)) == ["Hola mundo", "Prueba de voz"]


def test_reject_pipe_in_text(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text('texto\n"Frase con | símbolo"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="separadores"):
        studio.parse_texts(str(source))


def test_require_audio_for_clone():
    with pytest.raises(ValueError, match="referencia"):
        studio.check_reference(None, "", "Clonación por referencia")


def test_precise_clone_requires_transcript(tmp_path):
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"fake")
    with pytest.raises(ValueError, match="transcripción"):
        studio.check_reference(str(audio), "", "Clonación precisa")


def test_precise_clone_api(tmp_path):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"fake")
    calls = []
    fake = SimpleNamespace(
        tts_model=SimpleNamespace(sample_rate=24000),
        generate=lambda **kwargs: (calls.append(kwargs), np.zeros(2400))[1],
    )
    with patch.object(studio, "_model", fake):
        samples, sr = studio.synthesize(
            "Prueba", str(reference), "Voz de referencia",
            "Clonación precisa", 2.0, 10, 42,
        )
    assert sr == 24000 and len(samples) == 2400
    assert "seed" not in calls[0]
    assert calls[0]["prompt_text"] == "Voz de referencia"
    assert calls[0]["prompt_wav_path"] == str(reference)
    assert calls[0]["reference_wav_path"] == str(reference)


def test_dataset_zip_without_reference(tmp_path):
    source = tmp_path / "metadata.csv"
    source.write_text("texto\nHola mundo\nOtra frase\n", encoding="utf-8")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"private data")
    studio.OUTPUTS = tmp_path / "out"
    with patch.object(studio, "synthesize", return_value=(np.zeros(2400, dtype=np.float32), 24000)):
        zip_path, first_wav, report = studio.generate_dataset(
            str(source), str(reference), "Texto de referencia", "Clonación precisa",
            2.0, 10, 42, 22050,
            progress=lambda *a, **kw: None,
        )
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert names == {"metadata.csv", "report.json", "wavs/000001.wav", "wavs/000002.wav"}
        assert zf.read("metadata.csv").decode() == "000001|Hola mundo\n000002|Otra frase\n"
        assert b"private data" not in zf.read("report.json")
    assert first_wav.endswith("000001.wav")
    assert "2/2" in report


def test_whisper_requires_reference():
    from whisper_transcribe import transcribe_reference
    with pytest.raises(ValueError, match="audio de referencia"):
        transcribe_reference(None)


def test_gradio_ui_builds():
    view = studio.build_ui()
    assert len(view.blocks) >= 15


def test_launch_allows_only_the_studio_output_directory(monkeypatch, tmp_path):
    from unittest.mock import Mock
    studio.OUTPUTS = tmp_path / "voxcpm2_outputs"
    fake_ui = Mock()
    fake_ui.queue.return_value = fake_ui
    monkeypatch.setattr(studio, "build_ui", lambda: fake_ui)
    monkeypatch.setattr("sys.argv", ["studio.py"])
    monkeypatch.delenv("STUDIO_USER", raising=False)
    monkeypatch.delenv("STUDIO_PASSWORD", raising=False)
    studio.main()
    options = fake_ui.launch.call_args.kwargs
    assert options["allowed_paths"] == [str(studio.OUTPUTS.resolve())]
    assert options["share"] is False


def test_pyav_decodes_reference_audio(tmp_path):
    """Regression: PyAV 19 removed metadata_errors, breaking faster-whisper 1.2.1."""
    import soundfile as sf
    from faster_whisper.audio import decode_audio
    sound = tmp_path / "reference.wav"
    sf.write(str(sound), np.zeros(1600, dtype=np.float32), 16000)
    decoded = decode_audio(str(sound))
    assert decoded.shape == (1600,)


def test_clearing_reference_does_not_transcribe():
    from whisper_transcribe import transcribe_on_audio_change
    assert transcribe_on_audio_change(None) == ("", "")
    assert transcribe_on_audio_change("") == ("", "")
