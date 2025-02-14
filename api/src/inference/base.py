"""Base interface for Kokoro inference."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Tuple, Union

import numpy as np
import torch


class ModelBackend(ABC):
    """Abstract base class for model inference backend."""

    @abstractmethod
    async def load_model(self, path: str) -> None:
        """Load model from path.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio from text.

        Args:
            text: Input text to synthesize
            voice: Either a voice path or tuple of (name, tensor/path)
            speed: Speed multiplier

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Get device model is running on.

        Returns:
            Device string ('cpu' or 'cuda')
        """
        pass


class BaseModelBackend(ModelBackend):
    """Base implementation of model backend."""

    def __init__(self):
        """Initialize base backend."""
        self._model: Optional[torch.nn.Module] = None
        self._device: str = "cpu"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
