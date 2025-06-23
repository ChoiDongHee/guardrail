# ===================================================================
# 벡터 검색 디버깅 및 문제 해결 도구
# ===================================================================

import logging
import numpy as np
import json
from typing import Dict, Any, List, Optional
from guardrail_cache_v2.services.redis_service import EnhancedRedisManager, EmbeddingService

logger = logging.getLogger(__name__)


class VectorSearchDebugger:
    """벡터 검색 문제를 진단하고 해결하는 도구"""

    def __init__(self):
        self.redis_manager = EnhancedRedisManager.get_instance()
        self.embedding_service = EmbeddingService.get_instance()

    def debug_similarity_calculation(self, query_text: str, stored_question: str) -> Dict[str, Any]:
        """
        두 텍스트 간의 유사도 계산을 상세히 분석합니다.

        Args:
            query_text: 검색 쿼리
            stored_question: 저장된 질문

        Returns:
            상세한 디버깅 정보
        """
        debug_info = {
            "query_text": query_text,
            "stored_question": stored_question,
            "embeddings": {},
            "similarities": {},
            "potential_issues": []
        }

        try:
            # 1. 임베딩 생성
            query_embedding = self.embedding_service.get_embedding(query_text)
            stored_embedding = self.embedding_service.get_embedding(stored_question)

            debug_info["embeddings"] = {
                "query_shape": query_embedding.shape,
                "stored_shape": stored_embedding.shape,
                "query_norm": float(np.linalg.norm(query_embedding)),
                "stored_norm": float(np.linalg.norm(stored_embedding)),
                "query_mean": float(np.mean(query_embedding)),
                "stored_mean": float(np.mean(stored_embedding)),
                "query_std": float(np.std(query_embedding)),
                "stored_std": float(np.std(stored_embedding))
            }

            # 2. 다양한 유사도 계산
            # Cosine similarity
            cosine_sim = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )

            # Dot product
            dot_product = np.dot(query_embedding, stored_embedding)

            # Euclidean distance
            euclidean_dist = np.linalg.norm(query_embedding - stored_embedding)

            # Manhattan distance
            manhattan_dist = np.sum(np.abs(query_embedding - stored_embedding))

            debug_info["similarities"] = {
                "cosine_similarity": float(cosine_sim),
                "dot_product": float(dot_product),
                "euclidean_distance": float(euclidean_dist),
                "manhattan_distance": float(manhattan_dist),
                "cosine_distance": float(1.0 - cosine_sim)
            }

            # 3. Redis에서 실제 저장된 벡터와 비교
            redis_vector_info = self._check_redis_vector_integrity(stored_question)
            debug_info["redis_vector"] = redis_vector_info

            # 4. 잠재적 문제 진단
            debug_info["potential_issues"] = self._diagnose_issues(debug_info)

            return debug_info

        except Exception as e:
            debug_info["error"] = str(e)
            logger.error(f"디버깅 중 오류: {e}")
            return debug_info

    def _check_redis_vector_integrity(self, question: str) -> Dict[str, Any]:
        """Redis에 저장된 벡터의 무결성을 확인합니다."""
        try:
            redis_conn = self.redis_manager._get_redis_connection()
            redis_conn_raw = self.redis_manager._get_raw_redis_connection()

            # 텍스트 검색으로 해당 항목 찾기
            results = redis_conn.execute_command(
                "FT.SEARCH", self.redis_manager.index_name,
                f"@question:{question}",
                "RETURN", 2, "question", "question_vector",
                "LIMIT", 0, 1,
                "DIALECT", 2
            )

            if results[0] > 0:
                key = results[1]

                # 원시 벡터 데이터 가져오기
                vector_bytes = redis_conn_raw.hget(key, 'question_vector')

                if vector_bytes:
                    # bytes를 numpy array로 변환
                    vector_array = np.frombuffer(vector_bytes, dtype=np.float32)

                    return {
                        "found": True,
                        "key": key,
                        "vector_shape": vector_array.shape,
                        "vector_norm": float(np.linalg.norm(vector_array)),
                        "vector_mean": float(np.mean(vector_array)),
                        "vector_std": float(np.std(vector_array)),
                        "has_nan": bool(np.isnan(vector_array).any()),
                        "has_inf": bool(np.isinf(vector_array).any()),
                        "all_zeros": bool(np.allclose(vector_array, 0))
                    }

            return {"found": False, "reason": "항목을 찾을 수 없음"}

        except Exception as e:
            return {"found": False, "error": str(e)}

    def _diagnose_issues(self, debug_info: Dict[str, Any]) -> List[str]:
        """디버깅 정보를 바탕으로 잠재적 문제를 진단합니다."""
        issues = []

        # 임베딩 관련 이슈
        if "embeddings" in debug_info:
            emb_info = debug_info["embeddings"]

            # 벡터 크기 불일치
            if emb_info["query_shape"] != emb_info["stored_shape"]:
                issues.append(f"벡터 차원 불일치: {emb_info['query_shape']} vs {emb_info['stored_shape']}")

            # 벡터 정규화 문제
            if abs(emb_info["query_norm"] - 1.0) > 0.1:
                issues.append(f"쿼리 벡터가 정규화되지 않음: norm={emb_info['query_norm']:.4f}")

            if abs(emb_info["stored_norm"] - 1.0) > 0.1:
                issues.append(f"저장된 벡터가 정규화되지 않음: norm={emb_info['stored_norm']:.4f}")

            # 벡터 통계 이상
            if abs(emb_info["query_mean"]) > 0.5:
                issues.append(f"쿼리 벡터 평균값 이상: {emb_info['query_mean']:.4f}")

            if abs(emb_info["stored_mean"]) > 0.5:
                issues.append(f"저장된 벡터 평균값 이상: {emb_info['stored_mean']:.4f}")

        # 유사도 관련 이슈
        if "similarities" in debug_info:
            sim_info = debug_info["similarities"]

            # 코사인 유사도가 1.0에 너무 가까움
            if sim_info["cosine_similarity"] > 0.999:
                issues.append("코사인 유사도가 비정상적으로 높음 (거의 1.0)")

            # 유사하지 않은 텍스트임에도 높은 유사도
            cosine_sim = sim_info["cosine_similarity"]
            if cosine_sim > 0.8:
                issues.append(f"의미적으로 다른 텍스트임에도 높은 유사도: {cosine_sim:.4f}")

        # Redis 벡터 무결성 이슈
        if "redis_vector" in debug_info and debug_info["redis_vector"].get("found"):
            redis_info = debug_info["redis_vector"]

            if redis_info.get("has_nan"):
                issues.append("Redis 저장 벡터에 NaN 값 포함")

            if redis_info.get("has_inf"):
                issues.append("Redis 저장 벡터에 무한대 값 포함")

            if redis_info.get("all_zeros"):
                issues.append("Redis 저장 벡터가 모두 0")

        return issues

    def analyze_search_results(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """검색 결과를 상세히 분석합니다."""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_service.get_embedding(query)

            # Redis 검색 실행
            result = self.redis_manager.search_similar_question(query_embedding, limit)

            analysis = {
                "query": query,
                "total_results": result.get("total_results", 0),
                "similarity_threshold": self.redis_manager.similarity_threshold,
                "is_cache_hit": result.get("is_cache", False),
                "detailed_results": []
            }

            # 각 결과에 대한 상세 분석
            if result.get("total_results", 0) > 0:
                # Redis에서 전체 결과 다시 조회 (상세 분석용)
                vector_bytes = self.redis_manager._prepare_vector_bytes(query_embedding)
                redis_conn_raw = self.redis_manager._get_raw_redis_connection()

                query_args = [
                    "FT.SEARCH", self.redis_manager.index_name,
                    f"*=>[KNN {limit} @question_vector $BLOB]",
                    "PARAMS", 2, "BLOB", vector_bytes,
                    "RETURN", 7, "question", "response", "created_at", "last_accessed", "hits", "date_str",
                    "__question_vector_score",
                    "SORTBY", "__question_vector_score", "ASC",
                    "LIMIT", 0, limit,
                    "DIALECT", 2
                ]

                raw_results = redis_conn_raw.execute_command(*query_args)

                # 결과 파싱 및 분석
                for i in range(1, len(raw_results), 2):
                    if i + 1 < len(raw_results):
                        key = raw_results[i]
                        data = raw_results[i + 1]

                        # 데이터 파싱
                        data_iter = iter(data)
                        result_dict = {}
                        for k, v in zip(data_iter, data_iter):
                            result_dict[k] = v

                        # 거리를 유사도로 변환
                        distance = float(result_dict.get('__question_vector_score', 2.0))
                        similarity = max(0.0, 1.0 - (distance / 2.0)) if distance <= 2.0 else 0.0

                        # 개별 유사도 검증
                        stored_question = result_dict.get('question', '')
                        debug_result = self.debug_similarity_calculation(query, stored_question)

                        detailed_result = {
                            "rank": (i + 1) // 2,
                            "key": key,
                            "question": stored_question,
                            "redis_distance": distance,
                            "redis_similarity": similarity,
                            "manual_cosine_similarity": debug_result.get("similarities", {}).get("cosine_similarity"),
                            "potential_issues": debug_result.get("potential_issues", []),
                            "created_at": result_dict.get('created_at'),
                            "hits": result_dict.get('hits', 0)
                        }

                        analysis["detailed_results"].append(detailed_result)

            return analysis

        except Exception as e:
            logger.error(f"검색 결과 분석 중 오류: {e}")
            return {"error": str(e)}

    def check_model_consistency(self) -> Dict[str, Any]:
        """임베딩 모델의 일관성을 확인합니다."""
        test_texts = [
            "안녕하세요",
            "파이썬 프로그래밍",
            "퇴직연금",
            "주식투자",
            "데이터 분석"
        ]

        consistency_check = {
            "test_texts": test_texts,
            "embeddings_info": [],
            "pairwise_similarities": []
        }

        try:
            embeddings = []

            # 각 텍스트의 임베딩 생성
            for text in test_texts:
                embedding = self.embedding_service.get_embedding(text)
                embeddings.append(embedding)

                consistency_check["embeddings_info"].append({
                    "text": text,
                    "shape": embedding.shape,
                    "norm": float(np.linalg.norm(embedding)),
                    "mean": float(np.mean(embedding)),
                    "std": float(np.std(embedding))
                })

            # 쌍별 유사도 계산
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    cosine_sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )

                    consistency_check["pairwise_similarities"].append({
                        "text1": test_texts[i],
                        "text2": test_texts[j],
                        "cosine_similarity": float(cosine_sim)
                    })

            return consistency_check

        except Exception as e:
            consistency_check["error"] = str(e)
            return consistency_check

    def suggest_fixes(self, debug_info: Dict[str, Any]) -> List[str]:
        """문제점에 대한 해결책을 제안합니다."""
        suggestions = []

        issues = debug_info.get("potential_issues", [])

        for issue in issues:
            if "벡터 차원 불일치" in issue:
                suggestions.append("모든 벡터를 동일한 임베딩 모델로 재생성하세요.")

            elif "정규화되지 않음" in issue:
                suggestions.append("벡터를 L2 정규화하여 저장하세요.")

            elif "비정상적으로 높음" in issue:
                suggestions.append("벡터 저장 과정에서 중복이나 오류가 있는지 확인하세요.")

            elif "의미적으로 다른 텍스트임에도 높은 유사도" in issue:
                suggestions.append("다른 임베딩 모델을 사용하거나 전처리 과정을 점검하세요.")

            elif "NaN" in issue or "무한대" in issue:
                suggestions.append("벡터 생성 과정에서 수치적 오류를 확인하고 수정하세요.")

            elif "모두 0" in issue:
                suggestions.append("임베딩 생성이 실패했을 가능성이 있습니다. 모델과 입력을 확인하세요.")

        # 일반적인 권장사항
        suggestions.extend([
            "임계값(threshold)을 더 엄격하게 설정해보세요.",
            "키워드 전처리 과정을 점검하세요.",
            "Redis 인덱스를 재생성해보세요."
        ])

        return list(set(suggestions))  # 중복 제거


# ===================================================================
# 사용 예제 및 테스트
# ===================================================================

def main():
    """디버깅 도구 사용 예제"""
    debugger = VectorSearchDebugger()

    print("=" * 50)
    print("벡터 검색 디버깅 도구")
    print("=" * 50)

    # 1. 문제가 있었던 케이스 분석
    print("\n1. 문제 케이스 분석:")
    debug_result = debugger.debug_similarity_calculation("퇴직", "배당주")
    print(f"쿼리: {debug_result['query_text']}")
    print(f"저장된 질문: {debug_result['stored_question']}")
    print(f"코사인 유사도: {debug_result.get('similarities', {}).get('cosine_similarity', 'N/A'):.6f}")
    print(f"잠재적 문제: {debug_result.get('potential_issues', [])}")

    # 2. 검색 결과 상세 분석
    print("\n2. 검색 결과 분석:")
    search_analysis = debugger.analyze_search_results("퇴직", limit=5)
    print(f"총 {search_analysis.get('total_results', 0)}개 결과")
    print(f"캐시 히트: {search_analysis.get('is_cache_hit', False)}")

    for result in search_analysis.get("detailed_results", [])[:3]:
        print(f"  순위 {result['rank']}: '{result['question']}'")
        print(f"    Redis 유사도: {result['redis_similarity']:.6f}")
        print(f"    수동 계산 유사도: {result.get('manual_cosine_similarity', 'N/A')}")
        print(f"    문제점: {result.get('potential_issues', [])}")

    # 3. 모델 일관성 확인
    print("\n3. 모델 일관성 확인:")
    consistency = debugger.check_model_consistency()

    print("테스트 텍스트별 임베딩 정보:")
    for info in consistency.get("embeddings_info", []):
        print(f"  '{info['text']}': norm={info['norm']:.4f}, mean={info['mean']:.4f}")

    print("\n높은 유사도를 가진 쌍:")
    for sim in consistency.get("pairwise_similarities", []):
        if sim["cosine_similarity"] > 0.5:
            print(f"  '{sim['text1']}' vs '{sim['text2']}': {sim['cosine_similarity']:.4f}")

    # 4. 해결책 제안
    print("\n4. 권장 해결책:")
    suggestions = debugger.suggest_fixes(debug_result)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")


if __name__ == "__main__":
    main()