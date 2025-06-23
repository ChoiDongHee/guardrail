import os
import logging
import numpy as np
import time
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime

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
    _model = None

    @classmethod
    def get_instance(cls):
        """
        EmbeddingService의 싱글턴 인스턴스를 반환합니다.

        Returns:
            EmbeddingService: 싱글턴 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        싱글턴 인스턴스 생성 시 한 번만 실행되는 초기화 메서드입니다.
        환경 변수로부터 모델 이름을 가져와 SentenceTransformer 모델을 로드합니다.
        """
        # __init__이 여러 번 호출되는 것을 방지
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        try:
            logger.info("임베딩 서비스 초기화: Sentence-Transformers 모델 로딩 시작...")
            # 환경 변수에서 모델 이름 가져오기 (없으면 기본 모델 사용)
            model_name = os.getenv("EMBEDDING_MODEL", "snunlp/KR-SBERT-V40K-klueNLI-augSTS")
            # 모델 로드
            self._model = SentenceTransformer(model_name)
            logger.info(f"모델 '{model_name}' 로딩이 성공적으로 완료되었습니다.")


        except Exception as e:
            logger.error(f"임베딩 모델 로딩 중 심각한 오류 발생: {e}", exc_info=True)
            raise RuntimeError(f"임베딩 모델 로딩에 실패했습니다: {e}")

    def get_embedding(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        입력된 텍스트의 임베딩 벡터를 생성합니다.

        기본적으로 L2 정규화를 수행하여 코사인 유사도 검색에 바로 사용할 수 있는
        크기(norm)가 1인 벡터를 반환합니다.

        Args:
            text (str): 임베딩을 생성할 텍스트.
            normalize (bool): 벡터 정규화 수행 여부 (기본값: True).

        Returns:
            np.ndarray: 생성된 임베딩 벡터. 오류 발생 시 None을 반환합니다.
        """
        if not text or not isinstance(text, str):
            logger.warning("잘못된 입력(None 또는 str이 아님)으로 임베딩을 생성할 수 없습니다.")
            return None
        try:
            # 1. 모델을 사용하여 텍스트를 벡터로 인코딩
            embedding = self._model.encode(text)

            # 2. 벡터 정규화 (L2 Normalization) - 🔧 강화된 정규화
            embedding = self._normalize_vector_robust(embedding, text)

            # 3. 🔧 정규화 검증
            final_norm = np.linalg.norm(embedding)
            if normalize and abs(final_norm - 1.0) > 0.01:
                logger.warning(f"⚠️ 정규화 검증 실패: text='{text[:20]}...', norm={final_norm:.6f}")
            else:
                logger.debug(f"✅ 임베딩 생성 완료: text='{text[:20]}...', norm={final_norm:.6f}")

            return embedding

        except Exception as e:
            logger.error(f"'{text[:30]}...' 텍스트의 임베딩 생성 중 오류 발생: {e}", exc_info=True)
            return None

    def _normalize_vector_robust(self, vector: np.ndarray, text: str = "") -> np.ndarray:
        """
        강화된 벡터 정규화 함수

        Args:
            vector: 정규화할 벡터
            text: 디버깅용 텍스트 (선택사항)

        Returns:
            정규화된 벡터
        """
        try:
            # NaN이나 무한대 값 체크
            if np.isnan(vector).any():
                logger.error(f"❌ NaN 값이 포함된 벡터: '{text[:20]}...'")
                return vector

            if np.isinf(vector).any():
                logger.error(f"❌ 무한대 값이 포함된 벡터: '{text[:20]}...'")
                return vector

            # 벡터의 크기(L2 norm) 계산
            norm = np.linalg.norm(vector)

            if norm == 0:
                logger.warning(f"⚠️ 영벡터는 정규화할 수 없습니다: '{text[:20]}...'")
                return vector

            if norm < 1e-10:
                logger.warning(f"⚠️ 벡터 크기가 너무 작습니다: norm={norm:.2e}, text='{text[:20]}...'")
                return vector

            # 정규화 수행
            normalized = vector / norm

            # 정규화 결과 검증
            new_norm = np.linalg.norm(normalized)
            if abs(new_norm - 1.0) > 1e-6:
                logger.warning(f"⚠️ 정규화 정확도 문제: 예상=1.0, 실제={new_norm:.8f}, text='{text[:20]}...'")

            return normalized

        except Exception as e:
            logger.error(f"❌ 벡터 정규화 중 오류: {e}, text='{text[:20]}...'")
            return vector
