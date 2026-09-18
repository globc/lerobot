from typing import Any, Protocol, Union, runtime_checkable

import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage
from PIL import Image

IMG_SIZE = 256


@runtime_checkable
class TorchTensorLike(Protocol):
    def detach(self) -> "TorchTensorLike": ...

    def numpy(self) -> npt.NDArray[Any]: ...

    @property
    def is_cuda(self) -> bool: ...

    def cpu(self) -> "TorchTensorLike": ...


ImageTorch = TorchTensorLike

ImagePIL = PILImage.Image
ImageNumpyU8 = npt.NDArray[np.uint8]
ImageNumpyF32 = npt.NDArray[np.float32]
ImageNumpyF64 = npt.NDArray[np.float64]
ImageNumpy = Union[ImageNumpyU8, ImageNumpyF32, ImageNumpyF64]

ImageT = Union[ImagePIL, ImageNumpyU8, ImageNumpyF32, ImageNumpyF64, ImageTorch]

class ImageEncodingError(RuntimeError):
    """Raised when an image cannot be converted or encoded."""

    def __init__(self, message=None, **kwargs):
        if message is None:
            # Compose a default message from kwargs if available
            details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"Image encoding error. {details}" if details else "Image encoding error."
        super().__init__(message)
        self.details = kwargs

def normalize_numpy(image: ImageNumpy) -> ImageNumpy:
    """Normalize float arrays in [0,1] to uint8.

    Leaves non-float dtypes unchanged.
    """
    if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    return image

def to_numpy(image: ImageT) -> ImageNumpy:
    """Best-effort conversion to numpy array.

    Supports PIL.Image, numpy arrays, and torch-like tensors implementing
    ``detach`` & ``numpy``. Raises ImageEncodingError otherwise.
    """
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, ImagePIL):
        return np.array(image)
    if isinstance(image, TorchTensorLike):
        # Torch tensor path; guard against CUDA placement.
        if getattr(image, "is_cuda", False):
            image = image.cpu()
        return image.detach().numpy()
    raise ImageEncodingError(image_type=type(image))

def to_pil(image: ImageT) -> ImagePIL:
    """Convert image-like input to a resized PIL image.

    Accepted input types: PIL.Image.Image, numpy.ndarray, torch.Tensor-like.
    Handles (C,H,W) -> (H,W,C) channel-first conversion. Supports grayscale
    and RGB images. Raises ImageEncodingError on unsupported shapes.
    """
    if isinstance(image, ImagePIL):
        # Fast path for already-PIL images (only resizes)
        return image.resize((IMG_SIZE, IMG_SIZE))

    np_img = normalize_numpy(to_numpy(image))

    # Channel-first -> channel-last
    if np_img.ndim == 3 and np_img.shape[0] in (1, 3, 4):
        np_img = np.transpose(np_img, (1, 2, 0))

    if np_img.ndim == 2:  # grayscale
        pil = Image.fromarray(np_img, "L")  # single-channel
    elif np_img.ndim == 3:  # multi-channel
        if np_img.shape[2] == 3:  # RGB
            pil = Image.fromarray(np_img, "RGB")
        elif np_img.shape[2] == 1:  # grayscale with channel dim
            pil = Image.fromarray(np_img.squeeze(axis=2), "L")
        else:  # alpha or >4 channels not supported for now
            raise ImageEncodingError(message=f"Unsupported channel count: {np_img.shape[2]}")
    else:
        raise ImageEncodingError(image_shape=np_img.shape)

    return pil.resize((IMG_SIZE, IMG_SIZE))