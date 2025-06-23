# -----------------------------
# 📁 embedding_service.py (리팩토링 with getter/setter)
# -----------------------------
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime

# 환경 변수 로드
load_dotenv(override=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    텍스트 임베딩 생성을 담당하는 서비스 클래스입니다.
    벡터 정규화 기능이 포함된 싱글턴 패턴으로 구현됩니다.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        """싱글턴 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self._model_name = os.getenv("EMBEDDING_MODEL", "snunlp/KR-SBERT-V40K-klueNLI-augSTS")

        try:
            logger.info(f"임베딩 서비스 초기화: 모델 로딩 시작 ({self._model_name})")
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"모델 '{self._model_name}' 로딩 완료")
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 중 오류: {e}", exc_info=True)
            raise RuntimeError(f"임베딩 모델 로딩 실패: {e}")

    def get_embedding(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """입력 텍스트에 대한 임베딩 생성 및 정규화"""
        if not text or not isinstance(text, str):
            logger.warning("잘못된 입력: None 또는 문자열 아님")
            return None

        try:
            embedding = self._model.encode(text)
            embedding = self._normalize_vector_robust(embedding, text) if normalize else embedding

            final_norm = np.linalg.norm(embedding)
            if normalize and abs(final_norm - 1.0) > 0.01:
                logger.warning(f"⚠️ 정규화 검증 실패: text='{text[:20]}...', norm={final_norm:.6f}")
            else:
                logger.debug(f"✅ 임베딩 생성 완료: text='{text[:20]}...', norm={final_norm:.6f}")

            return embedding

        except Exception as e:
            logger.error(f"'{text[:30]}...' 텍스트의 임베딩 생성 오류: {e}", exc_info=True)
            return None

    def _normalize_vector_robust(self, vector: np.ndarray, text: str = "") -> np.ndarray:
        """강화된 벡터 정규화 함수"""
        try:
            if np.isnan(vector).any():
                logger.error(f"❌ NaN 포함 벡터: '{text[:20]}...'")
                return vector

            if np.isinf(vector).any():
                logger.error(f"❌ 무한대 포함 벡터: '{text[:20]}...'")
                return vector

            norm = np.linalg.norm(vector)

            if norm == 0:
                logger.warning(f"⚠️ 영벡터 정규화 불가: '{text[:20]}...'")
                return vector

            if norm < 1e-10:
                logger.warning(f"⚠️ 벡터 크기 너무 작음: norm={norm:.2e}, text='{text[:20]}...'")
                return vector

            normalized = vector / norm
            new_norm = np.linalg.norm(normalized)

            if abs(new_norm - 1.0) > 1e-6:
                logger.warning(f"⚠️ 정규화 정확도 문제: 기대=1.0, 실제={new_norm:.8f}, text='{text[:20]}...'")

            return normalized

        except Exception as e:
            logger.error(f"❌ 정규화 중 오류: {e}, text='{text[:20]}...'")

            return vector

    def encode_batch(self, texts: List[str], normalize: bool = True) -> List[np.ndarray]:
        """텍스트 리스트에 대한 일괄 임베딩 처리"""
        try:
            embeddings = self._model.encode(texts)
            return [self._normalize_vector_robust(v, t) if normalize else v for v, t in zip(embeddings, texts)]
        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {e}", exc_info=True)
            return []

    # --- Getter / Setter ---

    @property
    def model_name(self) -> str:
        """모델 이름 반환"""
        return self._model_name

    @model_name.setter
    def model_name(self, new_name: str):
        """모델 이름 변경 및 재로딩"""
        try:
            logger.info(f"모델 이름 변경 요청: {new_name}")
            self._model = SentenceTransformer(new_name)
            self._model_name = new_name
            logger.info(f"모델 재설정 완료: {new_name}")
        except Exception as e:
            logger.error(f"모델 재설정 실패: {e}", exc_info=True)

    def get_model(self):
        """내부 모델 인스턴스 반환"""
        return self._model
