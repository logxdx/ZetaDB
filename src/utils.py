import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize_vectors(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes vector to unit L2 norm (makes them length 1).

    Args:
        vector: A numpy array of vector (1D or 2D).

    Returns:
        The normalized vector. Returns the original vector if the norm is zero.
    """

    try:
        if vector.ndim == 1: # Single vector
            norm = np.linalg.norm(vector)
            if norm == 0:
                logger.warning("Cannot normalize a zero vector.")
                return vector
            return (vector.T / norm).T
        else:
            norms = np.linalg.norm(vector, axis=-1, keepdims=True)
            # Prevent division by zero for zero vector in the batch
            zero_norms_mask = (norms == 0)
            if np.any(zero_norms_mask):
                logger.warning(f"Found {np.sum(zero_norms_mask)} zero vector in batch during normalization. They will remain zero vector.")
                # Avoid division by zero, keep zero vector as they are.
                norms[zero_norms_mask] = 1e-10 # Replace norm with small value to avoid NaN, result will be ~0 vector
                # Or handle differently, e.g., return original vector for those rows?
                # For now, they become near-zero vector after division.

            return vector / norms
    except Exception as e:
        raise ValueError(f"Error normalizing vector: {e}") from e


def probe_media(path, modality: str) -> dict:
    """
    Best-effort media metadata used by the UI for layout and badges.

    Returns a dict with ``width``, ``height`` (pixels) and ``duration`` (seconds);
    missing values are 0.
    """
    info = {"width": 0, "height": 0, "duration": 0.0}
    try:
        if modality == "image":
            from PIL import Image

            with Image.open(path) as img:
                width, height = img.size
                # EXIF orientations 5-8 are rotated by 90° when displayed
                if img.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                    width, height = height, width
            info.update(width=width, height=height)
        elif modality == "video":
            from torchcodec.decoders import VideoDecoder

            meta = VideoDecoder(str(path)).metadata
            info.update(
                width=int(meta.width or 0),
                height=int(meta.height or 0),
                duration=float(meta.duration_seconds or 0.0),
            )
        elif modality == "audio":
            try:
                import soundfile

                info.update(duration=float(soundfile.info(str(path)).duration))
            except Exception:  # formats libsndfile can't read (e.g. m4a / aac)
                from torchcodec.decoders import AudioDecoder

                meta = AudioDecoder(str(path)).metadata
                info.update(duration=float(meta.duration_seconds or 0.0))
    except Exception as e:  # metadata is optional, never fail indexing because of it
        logger.debug(f"Could not probe {path}: {e}")
    return info
