"""VAE decode orchestration helpers for tiled latent-to-audio conversion."""

from typing import Optional

import torch
from loguru import logger

from acestep.gpu_config import get_effective_free_vram_gb


# Minimum free VRAM (GB) required to safely run GPU tiled decode when
# chunk_size has already been reduced to the smallest supported value.
# Below this threshold the VAE is moved to CPU for the decode pass so that
# the GPU decode does not deadlock on tight-memory systems (notably Windows
# WDDM where CUDA memory exhaustion can block instead of raising OOM).
_TILED_DECODE_MIN_FREE_VRAM_GB = 1.5


class VaeDecodeMixin:
    """High-level VAE decode entrypoints and fallback policies."""

    # MPS-safe chunk parameters (class-level for testability)
    _MPS_DECODE_CHUNK_SIZE = 32
    _MPS_DECODE_OVERLAP = 8

    def tiled_decode(
        self,
        latents,
        chunk_size: Optional[int] = None,
        overlap: int = 64,
        offload_wav_to_cpu: Optional[bool] = None,
    ):
        """Decode latents using tiling to reduce VRAM usage.

        Uses overlap-discard chunking to avoid boundary artifacts while
        constraining peak decode memory.

        Args:
            latents: Tensor shaped ``[batch, channels, latent_frames]``.
            chunk_size: Chunk size in latent frames. When ``None``, an
                auto-tuned value is selected by runtime policy.
            overlap: Overlap in latent frames between adjacent windows.
            offload_wav_to_cpu: Whether decoded waveform chunks should be
                offloaded to CPU immediately to reduce VRAM pressure.

        Returns:
            Decoded waveform tensor shaped ``[batch, audio_channels, samples]``.
        """
        # ---- MLX fast path (macOS Apple Silicon) ----
        if self.use_mlx_vae and self.mlx_vae is not None:
            try:
                result = self._mlx_vae_decode(latents)
                return result
            except Exception as exc:
                logger.warning(
                    f"[tiled_decode] MLX VAE decode failed ({type(exc).__name__}: {exc}), "
                    f"falling back to PyTorch VAE..."
                )

        # ---- PyTorch path (CUDA / MPS / CPU) ----
        if chunk_size is None:
            chunk_size = self._get_auto_decode_chunk_size()
        if offload_wav_to_cpu is None:
            offload_wav_to_cpu = self._should_offload_wav_to_cpu()

        logger.info(
            f"[tiled_decode] chunk_size={chunk_size}, offload_wav_to_cpu={offload_wav_to_cpu}, "
            f"latents_shape={latents.shape}"
        )

        # MPS Conv1d has a hard output-size limit during temporal upsampling.
        _is_mps = self.device == "mps"
        if _is_mps:
            _mps_chunk = self._MPS_DECODE_CHUNK_SIZE
            _mps_overlap = self._MPS_DECODE_OVERLAP
            _needs_reduction = (chunk_size > _mps_chunk) or (overlap > _mps_overlap)
            if _needs_reduction:
                logger.info(
                    f"[tiled_decode] VAE decode via PyTorch MPS; reducing chunk_size from {chunk_size} "
                    f"to {min(chunk_size, _mps_chunk)} and overlap from {overlap} "
                    f"to {min(overlap, _mps_overlap)} to avoid MPS conv output limit."
                )
                chunk_size = min(chunk_size, _mps_chunk)
                overlap = min(overlap, _mps_overlap)

        # When chunk_size is at its minimum value (VAE_DECODE_MAX_CHUNK_SIZE // 4,
        # set by MemoryUtilsMixin._get_auto_decode_chunk_size when free VRAM < 12 GB)
        # and available VRAM is critically low, proactively move the VAE to CPU to
        # prevent a CUDA deadlock/hang that can occur on Windows (WDDM) and other
        # tight-memory scenarios where CUDA stalls instead of raising
        # torch.cuda.OutOfMemoryError.
        # VAE_DECODE_MAX_CHUNK_SIZE is defined in MemoryUtilsMixin (memory_utils.py).
        _forced_cpu = False
        _forced_cpu_original_device = None
        _forced_cpu_original_dtype = None
        if chunk_size <= self.VAE_DECODE_MAX_CHUNK_SIZE // 4 and not _is_mps:
            try:
                from acestep.core.generation.handler.memory_utils import (
                    _cuda_device_index,
                    _is_cuda_device,
                )
                if _is_cuda_device(self.device):
                    free_gb = get_effective_free_vram_gb(_cuda_device_index(self.device))
                    if free_gb < _TILED_DECODE_MIN_FREE_VRAM_GB:
                        logger.warning(
                            f"[tiled_decode] VRAM critically low ({free_gb:.2f} GB) with "
                            f"min chunk_size={chunk_size}; moving VAE to CPU to prevent GPU hang"
                        )
                        _forced_cpu_original_device = next(self.vae.parameters()).device
                        _forced_cpu_original_dtype = self._get_vae_dtype(
                            str(_forced_cpu_original_device)
                        )
                        self._recursive_to_device(self.vae, "cpu", self._get_vae_dtype("cpu"))
                        latents = latents.cpu()
                        self._empty_cache()
                        _forced_cpu = True
                        # Waveform buffers will also land on CPU; explicit offload is
                        # unnecessary when the whole decode is on CPU already.
                        offload_wav_to_cpu = False
            except Exception:
                pass  # VRAM check is best-effort; proceed with GPU decode on failure

        try:
            result = self._tiled_decode_inner(latents, chunk_size, overlap, offload_wav_to_cpu)
        except (NotImplementedError, RuntimeError) as exc:
            if not _is_mps:
                raise
            logger.warning(
                f"[tiled_decode] MPS decode failed ({type(exc).__name__}: {exc}), "
                f"falling back to CPU VAE decode..."
            )
            result = self._tiled_decode_cpu_fallback(latents)
        finally:
            if _forced_cpu and _forced_cpu_original_device is not None:
                logger.info("[tiled_decode] Restoring VAE to GPU after CPU tiled decode")
                self._recursive_to_device(
                    self.vae, _forced_cpu_original_device, _forced_cpu_original_dtype
                )
                self._empty_cache()

        return result

    def _tiled_decode_cpu_fallback(self, latents):
        """Last-resort CPU VAE decode when MPS fails unexpectedly."""
        _first_param = next(self.vae.parameters())
        vae_device = _first_param.device
        vae_dtype = _first_param.dtype
        try:
            self.vae = self.vae.cpu().float()
            latents_cpu = latents.to(device="cpu", dtype=torch.float32)
            decoder_output = self.vae.decode(latents_cpu)
            result = decoder_output.sample
            del decoder_output
            return result
        finally:
            # Always restore VAE to original device/dtype
            self.vae = self.vae.to(vae_dtype).to(vae_device)

    def _decode_on_cpu(self, latents):
        """Move VAE to CPU, decode there, then restore original device."""
        logger.warning("[_decode_on_cpu] Moving VAE to CPU for decode (VRAM too tight for GPU decode)")

        try:
            original_device = next(self.vae.parameters()).device
        except StopIteration:
            original_device = torch.device("cpu")

        vae_cpu_dtype = self._get_vae_dtype("cpu")
        self._recursive_to_device(self.vae, "cpu", vae_cpu_dtype)
        self._empty_cache()

        latents_cpu = latents.cpu().to(vae_cpu_dtype)
        try:
            with torch.inference_mode():
                decoder_output = self.vae.decode(latents_cpu)
                result = decoder_output.sample
                del decoder_output
        finally:
            if original_device.type != "cpu":
                vae_gpu_dtype = self._get_vae_dtype(str(original_device))
                self._recursive_to_device(self.vae, original_device, vae_gpu_dtype)

        logger.info(f"[_decode_on_cpu] CPU decode complete, result shape={result.shape}")
        return result
