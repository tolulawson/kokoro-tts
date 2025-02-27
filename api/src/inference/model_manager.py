"""Kokoro V1 model management."""

from typing import Optional
import time
import os
import asyncio

from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1


class ModelManager:
    """Manages Kokoro V1 model loading and inference."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        self._backend: Optional[KokoroV1] = None  # Explicitly type as KokoroV1
        self._device: Optional[str] = None
        self._last_activity_time = time.time()
        self._idle_check_task = None
        
        # Get environment variables for idle timeout
        self._unload_model_after_idle = os.environ.get("UNLOAD_MODEL_AFTER_IDLE", "false").lower() == "true"
        try:
            self._idle_timeout_seconds = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "300"))
        except ValueError:
            self._idle_timeout_seconds = 300
            logger.warning("Invalid IDLE_TIMEOUT_SECONDS value, using default of 300 seconds")
        
        logger.info(f"Model unloading after idle: {self._unload_model_after_idle} (timeout: {self._idle_timeout_seconds}s)")

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        """Initialize Kokoro V1 backend."""
        try:
            self._device = self._determine_device()
            logger.info(f"Initializing Kokoro V1 on {self._device}")
            self._backend = KokoroV1()
            await self._start_idle_check()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kokoro V1: {e}")

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        """Initialize and warm up model.

        Args:
            voice_manager: Voice manager instance for warmup

        Returns:
            Tuple of (device, backend type, voice count)

        Raises:
            RuntimeError: If initialization fails
        """
        import time

        start = time.perf_counter()

        try:
            # Initialize backend
            await self.initialize()

            # Load model
            model_path = self._config.pytorch_kokoro_v1_file
            await self.load_model(model_path)

            # Use paths module to get voice path
            try:
                voices = await paths.list_voices()
                voice_path = await paths.get_voice_path(settings.default_voice)

                # Warm up with short text
                warmup_text = "Warmup text for initialization."
                # Use default voice name for warmup
                voice_name = settings.default_voice
                logger.debug(f"Using default voice '{voice_name}' for warmup")
                async for _ in self.generate(warmup_text, (voice_name, voice_path)):
                    pass
            except Exception as e:
                raise RuntimeError(f"Failed to get default voice: {e}")

            ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"Warmup completed in {ms}ms")

            return self._device, "kokoro_v1", len(voices)
        except FileNotFoundError as e:
            logger.error("""
Model files not found! You need to download the Kokoro V1 model:

1. Download model using the script:
   python docker/scripts/download_model.py --output api/src/models/v1_0

2. Or set environment variable in docker-compose:
   DOWNLOAD_MODEL=true
""")
            exit(0)
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    def get_backend(self) -> BaseModelBackend:
        """Get initialized backend.

        Returns:
            Initialized backend instance

        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        return self._backend

    async def load_model(self, path: str) -> None:
        """Load model using initialized backend.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If loading fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            await self._backend.load_model(path)
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    async def generate(self, *args, **kwargs):
        """Generate audio using initialized backend.

        Raises:
            RuntimeError: If generation fails
        """
        self._update_activity_timestamp()
        
        # Reload model if it was unloaded due to inactivity
        if self._backend is None or not self._backend.is_loaded:
            logger.info("Model was unloaded, reloading...")
            await self.initialize()
            model_path = self._config.pytorch_kokoro_v1_file
            await self.load_model(model_path)
        
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            async for chunk in self._backend.generate(*args, **kwargs):
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def unload_all(self) -> None:
        """Unload model and free resources."""
        if self._backend:
            self._backend.unload()
            self._backend = None

    @property
    def current_backend(self) -> str:
        """Get current backend type."""
        return "kokoro_v1"

    async def _start_idle_check(self):
        """Start background task to check for idle status and unload model if needed."""
        if self._idle_check_task is not None:
            self._idle_check_task.cancel()
        
        if not self._unload_model_after_idle:
            return
        
        async def check_idle():
            while True:
                await asyncio.sleep(60)  # Check every minute
                if self._backend and self._backend.is_loaded:
                    idle_time = time.time() - self._last_activity_time
                    if idle_time > self._idle_timeout_seconds:
                        logger.info(f"Model idle for {idle_time:.1f}s, unloading to free memory")
                        self.unload_all()
                    
        self._idle_check_task = asyncio.create_task(check_idle())
    
    def _update_activity_timestamp(self):
        """Update the last activity timestamp."""
        self._last_activity_time = time.time()


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override

    Returns:
        ModelManager instance
    """
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance
