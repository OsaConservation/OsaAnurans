"""Check actual bacpipe windowing and stable, immutable model-input audio."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from osa_anurans_app.src.audio import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    identify_recording,
    recording_windows,
    write_window_audio,
)


def test_bacpipe_resamples_mixes_and_wraps_the_complete_recording(tmp_path):
    from bacpipe import AudioHandler

    source_rate = 48_000
    time = np.arange(source_rate * 6, dtype=np.float32) / source_rate
    left = 0.2 * np.sin(2 * np.pi * 500 * time)
    right = 0.1 * np.sin(2 * np.pi * 800 * time)
    path = tmp_path / "six-seconds.wav"
    sf.write(path, np.stack([left, right], axis=1), source_rate, subtype="FLOAT")
    model = SimpleNamespace(
        sr=SAMPLE_RATE,
        segment_length=WINDOW_SAMPLES,
        model_name="perch_v2",
        only_embed_annotations=False,
        device="cpu",
        preprocess=lambda frames: frames,
    )
    handler = AudioHandler(model=model, audio_dir=tmp_path, padding="wrap")
    waveform, rate = handler.load_and_resample(path)
    frames = handler.prepare_audio(path).numpy()
    assert rate == SAMPLE_RATE
    assert frames.shape == (2, WINDOW_SAMPLES)
    assert np.array_equal(frames[0], waveform.numpy()[0, :WINDOW_SAMPLES])
    assert np.array_equal(frames[1, :SAMPLE_RATE], waveform.numpy()[0, WINDOW_SAMPLES:])
    assert np.array_equal(frames[1, SAMPLE_RATE:], waveform.numpy()[0, : 4 * SAMPLE_RATE])

    recording = identify_recording(path)
    windows = recording_windows(recording)
    assert len(windows) == 2
    assert windows[1].start_sample == 5 * source_rate
    assert windows[1].end_sample == 10 * source_rate
    for index, frame in enumerate(frames):
        output = tmp_path / f"window-{index}.wav"
        write_window_audio(output, frame)
        decoded, output_rate = sf.read(output, dtype="float32")
        assert output_rate == SAMPLE_RATE
        assert np.array_equal(decoded, frame)


def test_window_audio_is_immutable_and_ids_ignore_path_and_embedding_model(tmp_path):
    source = tmp_path / "source.wav"
    samples = np.linspace(-0.5, 0.5, WINDOW_SAMPLES, dtype=np.float32)
    write_window_audio(source, samples)
    identity = identify_recording(source)
    moved = tmp_path / "other-name.wav"
    moved.write_bytes(source.read_bytes())
    assert identify_recording(moved).recording_id == identity.recording_id
    assert recording_windows(replace(identity, path=moved)) == recording_windows(identity)

    write_window_audio(source, samples)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="waveform changed"):
        write_window_audio(source, samples * 0.5)
    assert source.read_bytes() == before


@pytest.mark.parametrize("source_rate", [22_050, 44_100, 48_000])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_resampled_windows_match_source_bounds_at_five_second_boundary(
    tmp_path, source_rate, offset
):
    from bacpipe import AudioHandler

    path = tmp_path / "boundary.wav"
    samples = np.linspace(-0.2, 0.2, source_rate * 5 + offset, dtype=np.float32)
    sf.write(path, samples, source_rate, subtype="FLOAT")
    model = SimpleNamespace(
        sr=SAMPLE_RATE,
        segment_length=WINDOW_SAMPLES,
        model_name="perch_v2",
        only_embed_annotations=False,
        device="cpu",
        preprocess=lambda frames: frames,
    )
    frames = AudioHandler(model=model, audio_dir=tmp_path, padding="wrap").prepare_audio(path)
    windows = recording_windows(identify_recording(path))
    assert frames.shape == (len(windows), WINDOW_SAMPLES)
    assert len(windows) == (2 if offset > 0 else 1)
    assert all(window.end_sample - window.start_sample == 5 * source_rate for window in windows)


def test_checkpoint_download_is_pinned_verified_and_retryable(tmp_path, monkeypatch):
    import huggingface_hub
    from osa_anurans_app.src import bacpipe_adapter

    content = b"verified checkpoint fixture"
    monkeypatch.setattr(bacpipe_adapter, "CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest())
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        checkpoint = kwargs["local_dir"] / kwargs["filename"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(content)
        return str(checkpoint)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    # Leftover files from an interrupted download must not suppress the retry.
    (tmp_path / "perch_v2").mkdir()
    (tmp_path / "perch_v2" / "incomplete.download").touch()
    checkpoint = bacpipe_adapter.ensure_checkpoint(tmp_path)
    assert calls[0]["revision"] == bacpipe_adapter.MODEL_REVISION
    assert bacpipe_adapter.ensure_checkpoint(tmp_path) == checkpoint
    assert len(calls) == 1
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        bacpipe_adapter.ensure_checkpoint(tmp_path)
