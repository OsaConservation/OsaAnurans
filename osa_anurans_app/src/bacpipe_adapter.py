"""A narrow, versioned boundary around bacpipe's Perch v2 embedder.

Responsibilities:
- Download/verify pinned Perch weights and select CPU or CUDA.
- Prepare five-second frames through bacpipe and predict embedding vectors.
- Describe the model/preprocessing configuration used for those vectors.

Audio rule:
- Feed the same prepared frames to inference and the local WAV writer.
"""

import hashlib
import json
import logging
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, WINDOW_POLICY, WINDOW_SAMPLES, file_hash

logger = logging.getLogger(__name__)
# Exact Hugging Face repository commit to download from; upstream updates cannot
# silently change the Perch weights used to generate our embeddings.
MODEL_REVISION = "f10c46a68eb9fcb18abe89394893400988248a70"

# Expected SHA-256 of the ONNX file. Verify downloaded/cached bytes before use;
# a mismatch stops processing. Also included in the embedding configuration ID.
CHECKPOINT_SHA256 = "4dcf71c18a147198545944bb5149697e89e3ad2e16637fa8f0edf6d13035a017"


def ensure_checkpoint(directory: Path) -> Path:
    """Return the local Perch ONNX path after verifying its contents.

    Steps:
    1. Download from MODEL_REVISION if the checkpoint is missing.
    2. Hash downloaded or cached bytes and compare with CHECKPOINT_SHA256.
    3. Stop on mismatch rather than load different/corrupt weights.
    """
    checkpoint = directory / "perch_v2" / "perch_v2_no_dft.onnx"
    if not checkpoint.is_file():
        from huggingface_hub import hf_hub_download

        logger.info("Downloading pinned Perch v2 checkpoint (414 MB)")
        hf_hub_download(
            repo_id="vskode/bacpipe_models",
            repo_type="dataset",
            filename="perch_v2/perch_v2_no_dft.onnx",
            revision=MODEL_REVISION,
            local_dir=directory,
        )
    if file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError(
            f"Perch checkpoint checksum mismatch: {checkpoint}; remove it and retry the download"
        )
    return checkpoint


class BacpipePerch:
    """Prepare audio and generate 1536-feature vectors with bacpipe's Perch model.

    Outputs:
    - prepare_audio: One row of samples per five-second window.
    - embed_frames: One vector per input row, preserving window order.
    - metadata/config_id: Identify the weights and preprocessing used.
    """

    def __init__(self, artifacts: Path, device: str = "auto", batch_size: int = 4):
        """Load verified weights and record the embedding configuration.

        Steps:
        1. Require the tested bacpipe version and choose an ONNX provider.
        2. Verify the checkpoint and initialize Perch with wrap padding.
        3. Check the model expects 32 kHz, five-second frames.
        4. Hash model/preprocessing metadata to produce config_id.

        Device rules:
        - auto: Prefer CUDA; permit CPU fallback.
        - cuda: Fail if CUDA cannot actually run the model.
        - cpu: Run inference on CPU.
        """
        import bacpipe
        import onnxruntime as ort

        if version("bacpipe") != "1.3.5":
            raise RuntimeError(
                "This adapter was verified with bacpipe 1.3.5; install the locked dependencies"
            )
        providers = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in providers
        if device == "cuda" and not has_cuda:
            raise RuntimeError("Perch CUDA requested but ONNX Runtime has no CUDA provider")
        selected = "cuda" if has_cuda and device != "cpu" else "cpu"
        if device == "auto" and selected == "cpu":
            logger.info("Perch uses CPU; no ONNX CUDA provider is available")

        checkpoints = artifacts / "checkpoints"
        ensure_checkpoint(checkpoints)
        self.embedder = bacpipe.Embedder(
            "perch_v2",
            audio_dir=str(artifacts),
            device=selected,
            model_base_path=checkpoints,
            padding="wrap",
            only_embed_annotations=False,
            run_pretrained_classifier=False,
            nr_parallel_workers=False,
        )
        # An installed CUDA provider may still fail to activate at runtime.
        active = self.embedder.model.model.active_providers
        if "CUDAExecutionProvider" not in active and selected == "cuda":
            if device == "cuda":
                raise RuntimeError("ONNX Runtime failed to activate the requested CUDA provider")
            logger.warning("ONNX Runtime fell back to CPU while initializing Perch")
        if (
            self.embedder.model.sr != SAMPLE_RATE
            or self.embedder.model.segment_length != WINDOW_SAMPLES
        ):
            raise RuntimeError(
                "Unexpected Perch input specification; window identity policy must be reviewed"
            )
        self.batch_size = batch_size
        checkpoint = checkpoints / "perch_v2" / "perch_v2_no_dft.onnx"
        self.metadata = {
            "model": "perch_v2",
            "bacpipe_version": version("bacpipe"),
            "checkpoint_sha256": file_hash(checkpoint),
            "sample_rate": SAMPLE_RATE,
            "window_samples": WINDOW_SAMPLES,
            "window_policy": WINDOW_POLICY,
            "resampling": "librosa-soxr_hq",
            "librosa_version": version("librosa"),
            "soxr_version": version("soxr"),
            "embedding_dimension": 1536,
        }
        serialized = json.dumps(self.metadata, sort_keys=True).encode()
        self.config_id = hashlib.sha256(serialized).hexdigest()[:24]
        logger.info("Perch providers: %s", self.embedder.model.model.active_providers)

    def prepare_audio(self, path: Path) -> np.ndarray:
        """Read/resample audio and return finite frames shaped (windows, 160000).

        Rules:
        - Use bacpipe's mono conversion, resampling, and wrap padding.
        - Return these exact frames for both inference and WAV writing.
        """
        frames = self.embedder.prepare_audio(path).detach().cpu().numpy()
        if frames.ndim != 2 or frames.shape[1] != WINDOW_SAMPLES or not np.isfinite(frames).all():
            raise ValueError(f"Invalid Perch window audio: {path}")
        return frames

    def embed_frames(self, frames: np.ndarray) -> np.ndarray:
        """Run Perch in batches and return a float32 matrix shaped (windows, 1536).

        Rules:
        - Preserve input order; row i describes frame i.
        - Disable gradients; this extracts vectors without training Perch.
        - Reject empty input or invalid output dimensions/values.
        """
        import torch

        results = []
        for start in range(0, len(frames), self.batch_size):
            batch = torch.from_numpy(
                np.asarray(frames[start : start + self.batch_size], dtype=np.float32)
            )
            with torch.no_grad():
                output = self.embedder.model(batch).detach().cpu().numpy()
            if output.ndim != 2 or output.shape[1] != 1536 or not np.isfinite(output).all():
                raise ValueError("Perch returned invalid embeddings")
            results.append(output)
        if not results:
            raise ValueError("Cannot embed an empty recording")
        return np.concatenate(results).astype(np.float32, copy=False)
