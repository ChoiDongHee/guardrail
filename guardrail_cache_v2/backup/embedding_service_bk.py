import os
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(override=True)  # 강제 재로드

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    Implemented as a singleton.
    """
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of EmbeddingService.
        Returns:
            EmbeddingService: Singleton instance
        """
        if cls._instance is None:
            cls._instance = EmbeddingService()
        return cls._instance

    def __init__(self):
        """Initialize embedding service with the specified model."""
        if EmbeddingService._model is None:
            try:
                logger.info("embedding_service.py:__init__:step1 - Loading sentence-transformers model")
                # Get model name from environment or use default
                model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                # Load model
                EmbeddingService._model = SentenceTransformer(model_name)
                logger.info(f"embedding_service.py:__init__:step2 - Model '{model_name}' loaded successfully")
            except Exception as e:
                logger.error(f"embedding_service.py:__init__:step3 - Error loading model: {e}")
                raise RuntimeError(f"Failed to load embedding model: {e}")

    def get_embedding(self, text):
        """
        Generate embedding vector for input text.

        Args:
            text (str): Input text to embed

        Returns:
            numpy.ndarray: Embedding vector
        """
        try:
            if not text or not isinstance(text, str):
                logger.warning("embedding_service.py:get_embedding - Invalid input text")
                return None

            embedding = EmbeddingService._model.encode(text)
            return embedding

        except Exception as e:
            logger.error(f"embedding_service.py:get_embedding - Error generating embedding: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")