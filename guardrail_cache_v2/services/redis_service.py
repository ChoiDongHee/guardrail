# ===================================================================
# 파일: enhanced_redis_util.py
# 설명: 향상된 Redis 검색 및 캐싱 시스템
# ===================================================================

import os
import logging
import time
import json
import numpy as np
from datetime import datetime
from redis import ConnectionPool, Redis
from redis.exceptions import ResponseError, RedisError
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
load_dotenv(override=True)  # 강제 재로드


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedRedisManager:
    """향상된 Redis 연결 및 데이터 관리를 위한 싱글턴 클래스"""
    _instance = None

    @classmethod
    def get_instance(cls):
        """싱글턴 인스턴스를 가져옵니다."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        인스턴스가 없을 때만 실행되는 초기화 메서드.
        Redis 연결 풀을 생성하고 인덱스를 설정합니다.
        """
        # 싱글턴 패턴: __init__이 여러 번 호출되는 것을 방지
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        logger.info("EnhancedRedisManager 인스턴스를 초기화합니다...")

        # 환경 변수에서 Redis 연결 정보 가져오기
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.db = int(os.getenv("REDIS_DB", 0))
        self.password = os.getenv("REDIS_PASSWORD", None)

        # 연결 풀 생성 (애플리케이션 전체에서 공유)
        self.pool = ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True  # 자동 UTF-8 디코딩 활성화
        )

        # 설정값 초기화
        self.index_name = os.getenv("REDIS_INDEX_NAME", "chat_admin_idx")
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "chat_data:")
        self.vector_dim = int(os.getenv("VECTOR_DIM", 768))
        self.similarity_threshold = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", 0.95))

        # 인덱스 생성 확인
        self._ensure_index_exists()

    def _get_redis_connection(self) -> Redis:
        """연결 풀에서 Redis 연결을 가져옵니다. (자동 디코딩 활성화)"""
        return Redis(connection_pool=self.pool)

    def _get_raw_redis_connection(self) -> Redis:
        """연결 풀에서 원시(bytes) Redis 연결을 가져옵니다. (벡터 데이터용)"""
        return Redis(connection_pool=self.pool, decode_responses=False)

    def _ensure_index_exists(self) -> bool:
        """인덱스가 존재하는지 확인하고, 없으면 생성합니다."""
        try:
            redis_conn = self._get_redis_connection()

            # 인덱스 존재 여부 확인
            if self._index_exists():
                logger.info(f"인덱스 '{self.index_name}'가 이미 존재합니다.")
                return True

            # 인덱스 생성
            return self._create_search_index()

        except Exception as e:
            logger.error(f"인덱스 확인 중 오류 발생: {e}")
            return False

    def _index_exists(self) -> bool:
        """인덱스가 존재하는지 확인합니다."""
        try:
            redis_conn = self._get_redis_connection()
            # FT.INFO 명령으로 인덱스 존재 여부 확인
            redis_conn.execute_command("FT.INFO", self.index_name)
            return True
        except ResponseError:
            # 인덱스가 없으면 ResponseError 발생
            return False
        except Exception as e:
            logger.error(f"인덱스 존재 여부 확인 중 오류: {e}")
            return False

    def _create_search_index(self) -> bool:
        """향상된 검색 인덱스를 생성합니다."""
        try:
            redis_conn = self._get_redis_connection()

            # 향상된 스키마 정의
            schema = [
                # 벡터 검색 필드
                'question_vector', 'VECTOR', 'HNSW', '6',
                'TYPE', 'FLOAT32',
                'DIM', str(self.vector_dim),
                'DISTANCE_METRIC', 'COSINE',

                # 텍스트 검색 필드
                'question', 'TEXT', 'SORTABLE',
                'response', 'TEXT',

                # 수치 및 정렬 필드
                'created_at', 'NUMERIC', 'SORTABLE',
                'last_accessed', 'NUMERIC', 'SORTABLE',
                'hits', 'NUMERIC', 'SORTABLE',

                # 날짜 태그 필드 (YYYYMMDD 형식)
                'date_str', 'TAG', 'SORTABLE'
            ]

            cmd = [
                'FT.CREATE', self.index_name,
                'ON', 'HASH',
                'PREFIX', '1', self.key_prefix,
                'SCHEMA', *schema
            ]

            redis_conn.execute_command(*cmd)
            logger.info(f"인덱스 '{self.index_name}'를 성공적으로 생성했습니다.")
            return True

        except ResponseError as e:
            if "Index already exists" in str(e):
                logger.info(f"인덱스 '{self.index_name}'는 이미 존재합니다.")
                return True
            else:
                logger.error(f"인덱스 생성 중 Redis 오류: {e}")
                return False
        except Exception as e:
            logger.error(f"인덱스 생성 중 예상치 못한 오류: {e}")
            return False

    # --- 벡터 검색 관련 헬퍼 함수들 ---

    def _prepare_vector_bytes(self, vector) -> Optional[bytes]:
        """입력 벡터를 검증하고 Redis가 요구하는 bytes 형식으로 변환합니다."""
        try:
            if isinstance(vector, bytes):
                return vector
            if hasattr(vector, 'shape') or isinstance(vector, (list, tuple)):
                return np.array(vector, dtype=np.float32).tobytes()

            logger.error(f"지원하지 않는 벡터 타입입니다: {type(vector)}")
            return None
        except Exception as e:
            logger.error(f"벡터 변환 중 오류 발생: {e}")
            return None

    def _build_vector_search_query(self, vector_bytes: bytes, limit: int = 5) -> List:
        """벡터 검색 쿼리를 구성합니다."""
        return [
            "FT.SEARCH", self.index_name,
            f"*=>[KNN {limit} @question_vector $BLOB]",
            "PARAMS", 2, "BLOB", vector_bytes,
            "RETURN", 7, "question", "response", "created_at", "last_accessed", "hits", "date_str",
            "__question_vector_score",
            "SORTBY", "__question_vector_score", "ASC",  # 거리순 정렬 추가
            "LIMIT", 0, limit,
            "DIALECT", 2
        ]

    def _parse_search_results(self, results: List, apply_threshold: bool = True) -> Dict[str, Any]:
        """검색 결과를 파싱하고 임계값을 적용합니다."""
        response_data = {
            "is_cache": False,
            "similarity": 0.0,
            "data": {},
            "total_results": 0
        }

        if not results or results[0] == 0:
            logger.info("❌ 검색 결과가 없습니다.")
            return response_data

        response_data["total_results"] = results[0]
        logger.info(f"📊 전체 검색 결과 수: {results[0]}")

        # 모든 결과 로그 출력 (디버깅용)
        logger.info("🔍 모든 검색 결과:")
        for i in range(1, len(results), 2):
            if i + 1 < len(results):
                key = results[i]
                data = results[i + 1]
                # 간단한 데이터 파싱
                data_iter = iter(data)
                temp_dict = {}
                for k, v in zip(data_iter, data_iter):
                    temp_dict[k] = v

                distance = float(temp_dict.get('__question_vector_score', 2.0))
                question = temp_dict.get('question', '')[:30]
                logger.info(f"   결과 {(i + 1) // 2}: 거리={distance:.6f}, 질문='{question}...'")

        # 첫 번째 결과 처리 (정렬 후 가장 유사한 결과)
        if len(results) >= 3:
            try:
                key = results[1]
                data = results[2]

                # 결과 데이터 파싱
                data_iter = iter(data)
                result_dict = {}
                for k, v in zip(data_iter, data_iter):
                    result_dict[k] = v

                # 유사도 계산
                distance = float(result_dict.get('__question_vector_score', 2.0))

                # COSINE 거리를 유사도로 변환
                # 거리: 0(완전 동일) ~ 2(완전 다름)
                # 유사도: 1(완전 동일) ~ 0(완전 다름)
                if distance <= 2.0:
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                else:
                    similarity = 0.0

                response_data["similarity"] = round(similarity, 4)

                logger.info(f"🎯 최적 결과 - 거리값: {distance:.6f} → 유사도: {similarity:.4f} ({similarity * 100:.1f}%)")
                logger.info(f"📝 매칭된 질문: '{result_dict.get('question', '')}'")

                # 임계값 적용
                if apply_threshold and similarity >= self.similarity_threshold:
                    response_data["is_cache"] = True

                    # 결과 데이터 구성
                    response_data["data"] = {
                        "id": key.replace(self.key_prefix, ''),
                        "question": result_dict.get('question', ''),
                        "response": result_dict.get('response', ''),
                        "created_at": int(result_dict.get('created_at', 0)),
                        "last_accessed": int(result_dict.get('last_accessed', 0)),
                        "hits": int(result_dict.get('hits', 0))
                    }

                    logger.info(f"✅ 캐시 히트! 유사도 {similarity:.4f} >= 임계값 {self.similarity_threshold}")
                    logger.info(f"📄 반환할 답변: '{result_dict.get('response', '')[:50]}...'")
                else:
                    logger.info(f"❌ 유사도 임계값 미달: {similarity:.4f} < {self.similarity_threshold}")
                    logger.info(f"💡 힌트: 임계값을 {similarity:.2f} 이하로 낮추면 히트됩니다.")

            except Exception as e:
                logger.error(f"❌ 검색 결과 파싱 중 오류: {e}", exc_info=True)

        return response_data
    # --- 주요 검색 및 캐싱 메서드 ---

    def search_similar_question(self, question_vector, limit: int = 5) -> Dict[str, Any]:
        """
        유사한 질문을 검색하고 캐시 히트 여부를 판단합니다.

        Args:
            question_vector: 질문의 임베딩 벡터
            limit: 검색할 최대 결과 수

        Returns:
            Dict: 검색 결과 및 캐시 히트 정보
        """
        start_time = time.time()

        try:
            # 벡터 변환
            vector_bytes = self._prepare_vector_bytes(question_vector)
            if not vector_bytes:
                logger.error("벡터 변환에 실패했습니다.")
                return {"is_cache": False, "similarity": 0.0, "data": {}, "total_results": 0}

            # 검색 쿼리 구성 및 실행
            query_args = self._build_vector_search_query(vector_bytes, limit)
            redis_conn_raw = self._get_raw_redis_connection()

            logger.info(f"벡터 검색 시작: limit={limit}")
            results = redis_conn_raw.execute_command(*query_args)
            logger.info(f"결과: limit={results}")
            # 결과 파싱
            parsed_result = self._parse_search_results(results)

            # 캐시 히트 시 접근 기록 업데이트
            if parsed_result["is_cache"] and parsed_result["data"]:
                self._update_access_record(parsed_result["data"]["id"])
            elapsed_time = time.time() - start_time
            logger.info(f"벡터 검색 완료: {elapsed_time:.3f}초")

            return parsed_result

        except RedisError as e:
            logger.error(f"[벡터 검색] Redis 쿼리 실행 중 오류 발생: {e}", exc_info=True)
            return {"is_cache": False, "similarity": 0.0, "data": {}, "total_results": 0}
        except Exception as e:
            logger.error(f"[벡터 검색] 예상치 못한 오류: {e}", exc_info=True)
            return {"is_cache": False, "similarity": 0.0, "data": {}, "total_results": 0}

    def _update_access_record(self, entry_id: str) -> bool:
        """접근 기록을 업데이트합니다 (hits 증가, last_accessed 갱신)."""
        try:
            redis_conn = self._get_redis_connection()
            key = f"{self.key_prefix}{entry_id}"

            if not redis_conn.exists(key):
                logger.warning(f"[접근 기록 업데이트] 항목을 찾을 수 없음: id={entry_id}")
                return False

            # 현재 데이터 가져오기
            current_data = redis_conn.hmget(key, ['hits', 'last_accessed'])

            current_hits = int(current_data[0] if current_data[0] else 0)
            current_time = int(time.time())

            # 변경 전 (Deprecated):
            # redis_conn.hmset(key, {
            #     'hits': current_hits + 1,
            #     'last_accessed': current_time
            # })

            # 변경 후 (권장):
            redis_conn.hset(key, mapping={
                'hits': current_hits + 1,
                'last_accessed': current_time
            })

            logger.info(f"[접근 기록 업데이트] 완료: id={entry_id}, hits={current_hits + 1}")
            return True

        except Exception as e:
            logger.error(f"[접근 기록 업데이트] 오류: {e}")
            return False

    def store_question_response(self, question: str, response: str, question_vector) -> Optional[str]:
        """질문과 응답 쌍을 Redis에 저장합니다."""
        try:
            # 고유 ID 생성
            timestamp = int(time.time())
            entry_id = f"{timestamp}_{hash(question) % 10000}"
            key = f"{self.key_prefix}{entry_id}"

            # 벡터 변환
            vector_bytes = self._prepare_vector_bytes(question_vector)
            if not vector_bytes:
                raise ValueError("유효하지 않은 벡터로 변환 불가")

            # 날짜 문자열 생성 (YYYYMMDD)
            date_str = datetime.now().strftime("%Y%m%d")

            # 데이터 구성 (자동 인코딩을 위해 문자열로 저장)
            data_fields = {
                'question': question,
                'response': response,
                'created_at': str(timestamp),
                'last_accessed': str(timestamp),
                'hits': '0',
                'date_str': date_str
            }

            # Redis에 저장
            redis_conn = self._get_redis_connection()
            redis_conn_raw = self._get_raw_redis_connection()

            # 파이프라인 사용
            pipe = redis_conn.pipeline()

            # 텍스트 데이터는 자동 인코딩 연결로 저장
            pipe.hset(key, mapping=data_fields)
            pipe.execute()

            # 벡터 데이터는 raw 연결로 저장
            redis_conn_raw.hset(key, 'question_vector', vector_bytes)

            logger.info(f"[저장] 항목 저장 완료: id={entry_id}")
            return entry_id

        except (RedisError, ValueError) as e:
            logger.error(f"[저장] 항목 저장 중 오류: {e}", exc_info=True)
            return None

    def get_entry_by_id(self, entry_id: str, update_access: bool = True) -> Optional[Dict[str, Any]]:
        """ID로 특정 항목을 조회합니다."""
        try:
            redis_conn = self._get_redis_connection()
            key = f"{self.key_prefix}{entry_id}"

            if not redis_conn.exists(key):
                logger.warning(f"[항목 조회] 항목을 찾을 수 없음: id={entry_id}")
                return None

            # 모든 필드 가져오기
            data = redis_conn.hgetall(key)
            if not data:
                return None

            # 결과 구성
            entry_data = {
                "id": entry_id,
                "question": data.get('question', ''),
                "response": data.get('response', ''),
                "created_at": int(data.get('created_at', 0)),
                "last_accessed": int(data.get('last_accessed', 0)),
                "hits": int(data.get('hits', 0)),
                "date_str": data.get('date_str', '')
            }

            # 접근 기록 업데이트
            if update_access:
                current_time = int(time.time())
                new_hits = entry_data['hits'] + 1

                redis_conn.hset(key, mapping={
                    'hits': new_hits,
                    'last_accessed': current_time
                })

                entry_data['hits'] = new_hits
                entry_data['last_accessed'] = current_time

            logger.info(f"[항목 조회] 완료: id={entry_id}, hits={entry_data['hits']}")
            return entry_data

        except Exception as e:
            logger.error(f"[항목 조회] ID {entry_id} 조회 중 오류: {e}", exc_info=True)
            return None

    def get_redis_json_key(self,key: str):
        try:
            redis_conn = self._get_redis_connection()
            raw = redis_conn.get(key)
            if raw:
                return json.loads(raw)
            else:
                logger.error(f"⚠️ Redis에서 키 '{key}'의 값을 찾을 수 없습니다.")
                return []
        except Exception as e:
            logger.error(f"❌ Redis에서 키 '{key}' 로드 실패: {e}")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보를 반환합니다."""
        try:
            redis_conn = self._get_redis_connection()

            # 전체 항목 수 조회
            results = redis_conn.execute_command(
                "FT.SEARCH", self.index_name, "*",
                "LIMIT", 0, 0  # 개수만 확인
            )

            total_count = results[0] if results else 0

            # 인덱스 정보 가져오기
            try:
                index_info = redis_conn.execute_command("FT.INFO", self.index_name)
                index_size = dict(zip(index_info[::2], index_info[1::2]))
            except:
                index_size = {}

            return {
                "total_entries": total_count,
                "index_name": self.index_name,
                "vector_dimension": self.vector_dim,
                "similarity_threshold": self.similarity_threshold,
                "index_info": index_size
            }

        except Exception as e:
            logger.error(f"캐시 통계 조회 중 오류: {e}")
            return {"error": str(e)}

    def delete_entry(self, entry_id: str) -> bool:
        """ID로 항목을 삭제합니다."""
        try:
            redis_conn = self._get_redis_connection()
            key = f"{self.key_prefix}{entry_id}"

            result = redis_conn.delete(key)
            if result > 0:
                logger.info(f"[삭제] 항목 삭제 완료: id={entry_id}")
                return True
            else:
                logger.warning(f"[삭제] 항목을 찾을 수 없음: id={entry_id}")
                return False

        except RedisError as e:
            logger.error(f"[삭제] ID {entry_id} 삭제 중 오류: {e}", exc_info=True)
            return False


# EmbeddingService 클래스 (기존 코드 참조)
class EmbeddingService:
    """임베딩 생성 서비스 (싱글턴)"""
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if EmbeddingService._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("임베딩 모델 로딩 중...")
                model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                EmbeddingService._model = SentenceTransformer(model_name)
                logger.info(f"임베딩 모델 '{model_name}' 로딩 완료")
            except Exception as e:
                logger.error(f"임베딩 모델 로딩 실패: {e}")
                raise RuntimeError(f"임베딩 모델 로딩 실패: {e}")

    def get_embedding(self, text: str):
        """텍스트의 임베딩 벡터를 생성합니다."""
        try:
            if not text or not isinstance(text, str):
                logger.warning("잘못된 입력 텍스트")
                return None

            embedding = EmbeddingService._model.encode(text)
            return embedding

        except Exception as e:
            logger.error(f"임베딩 생성 중 오류: {e}")
            raise RuntimeError(f"임베딩 생성 실패: {e}")


# 통합 캐시 서비스 클래스
class QuestionAnswerCacheService:
    """질문-답변 캐시 서비스 통합 클래스"""

    def __init__(self):
        self.redis_manager = EnhancedRedisManager.get_instance()
        self.embedding_service = EmbeddingService.get_instance()

    def search_cached_answer(self, question: str) -> Dict[str, Any]:
        """질문에 대한 캐시된 답변을 검색합니다."""
        try:
            # 질문의 임베딩 생성
            question_embedding = self.embedding_service.get_embedding(question)
            if question_embedding is None:
                return {"is_cache": False, "error": "임베딩 생성 실패"}

            # 유사한 질문 검색
            result = self.redis_manager.search_similar_question(question_embedding)

            return result

        except Exception as e:
            logger.error(f"캐시 검색 중 오류: {e}")
            return {"is_cache": False, "error": str(e)}

    def cache_question_answer(self, question: str, answer: str) -> Optional[str]:
        """질문-답변 쌍을 캐시에 저장합니다."""
        try:
            # 질문의 임베딩 생성
            question_embedding = self.embedding_service.get_embedding(question)
            if question_embedding is None:
                logger.error("임베딩 생성 실패")
                return None

            # Redis에 저장
            entry_id = self.redis_manager.store_question_response(
                question, answer, question_embedding
            )

            return entry_id

        except Exception as e:
            logger.error(f"캐시 저장 중 오류: {e}")
            return None


# 사용 예제
def main():
    """사용 예제"""
    try:
        # 캐시 서비스 초기화
        cache_service = QuestionAnswerCacheService()

        # 질문-답변 저장
        question = "Python에서 리스트를 정렬하는 방법은?"
        answer = "Python에서 리스트를 정렬하려면 sort() 메서드나 sorted() 함수를 사용할 수 있습니다."

        entry_id = cache_service.cache_question_answer(question, answer)
        print(f"저장된 항목 ID: {entry_id}")

        # 유사한 질문 검색
        similar_question = "파이썬 리스트 정렬 방법"
        result = cache_service.search_cached_answer(similar_question)

        print(f"검색 결과: {json.dumps(result, indent=2, ensure_ascii=False)}")

        # 캐시 통계
        stats = cache_service.redis_manager.get_cache_stats()
        print(f"캐시 통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")

        result = cache_service.search_cached_answer("Python에서 리스트를 정렬하는 방법은?")
        logger.debug("==   검색 결과 ==")
        logger.debug(result)
        logger.debug("==   검색 결과 ==")

    except Exception as e:
        logger.error(f"예제 실행 중 오류: {e}")


if __name__ == "__main__":
    main()