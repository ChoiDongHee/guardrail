import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pykospacing import Spacing

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import uuid


from guardrail_cache_v2.services.mecapKoreanAnalyzer_service import RefinedKoreanPreprocessor
from guardrail_cache_v2.services.redis_service import EnhancedRedisManager
from guardrail_cache_v2.services.embedding_service import EmbeddingService


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Redis Vector Similarity Search API", version="1.0.0")


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (CSS, JS)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# ✅ 서비스 초기화
# -----------------------------
redis_client = EnhancedRedisManager()
SYNONYM_DICTS = redis_client.get_redis_json_key("SYNONYM_DICTS")
print("============================================================")
print(SYNONYM_DICTS)
print("============================================================")
processor = RefinedKoreanPreprocessor(synonym_dict=SYNONYM_DICTS)
rep_map = processor.build_synonym_map()
embedding_service = EmbeddingService.get_instance()


# -----------------------------
# ✅ Pydantic 모델
# -----------------------------
class QAItem(BaseModel):
    question: str
    method: Optional[str] = "vector"  # "vector" or "keyword"
    answer: str


class SearchQuery(BaseModel):
    query: str
    method: Optional[str] = "vector"  # 기본값: vector ("vector" or "keyword")


class SearchResponse(BaseModel):
    state: bool
    cached: bool
    similarity: float
    fresh_data: Optional[Dict] = None
    error: Optional[str] = None


class RegisterResponse(BaseModel):
    state: bool
    message: str
    id: Optional[str] = None
    error: Optional[str] = None


# -----------------------------
# ✅ HTML 렌더링 라우터
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("templates/index.html", encoding="utf-8") as f:
            content = f.read()
            # 템플릿 변수 치환
            content = content.replace("{{ similarity_threshold }}", str(redis_client.similarity_threshold))
            return HTMLResponse(content=content)
    except Exception as e:
        logger.error("❌ 메인 페이지 로딩 실패", exc_info=e)
        raise HTTPException(status_code=500, detail="메인 페이지 로딩 실패")


@app.get("/admin", response_class=HTMLResponse)
async def get_admin():
    try:
        with open("templates/admin.html", encoding="utf-8") as f:
            content = f.read()
            return HTMLResponse(content=content)
    except Exception as e:
        logger.error("❌ 관리자 페이지 로딩 실패", exc_info=e)
        raise HTTPException(status_code=500, detail="관리자 페이지 로딩 실패")


# -----------------------------
# ✅ API 엔드포인트
# -----------------------------

@app.post("/api/query", response_model=SearchResponse)
async def search_question(search_query: SearchQuery):
    """
    질문 검색 API - 두 가지 방식 지원
    - vector: 전체 질문을 벡터로 변환하여 유사도 검색
    - keyword: MeCab으로 키워드 추출 후 검색
    """
    try:
        logger.info(f"🔍 검색 요청: method={search_query.method}, query={search_query.query}")
        # # 가장 먼저, 다른 어떤 임포트보다도 먼저 환경변수 설정
        spacing = Spacing()
        search_query.query = spacing(search_query.query)
        logger.info(f"🔍  띄어쓰기 검사기: method={search_query.method}, query={search_query.query}")

        if search_query.method == "keyword":
            # 키워드 기반 검색
            return await _search_by_keywords(search_query.query)
        else:
            # 벡터 기반 검색 (기본값)
            return await _search_by_vector(search_query.query)

    except Exception as e:
        logger.error("❌ 검색 실패", exc_info=e)
        return SearchResponse(
            state=False,
            cached=False,
            similarity=0.0,
            error=str(e)
        )


@app.post("/api/data", response_model=RegisterResponse)
async def register_question(qa: QAItem):
    """
    질문-답변 등록 API
    - vector: 질문 전체를 임베딩
    - keyword: MeCab 처리된 키워드를 임베딩
    """
    try:
        logger.info(f"📝 질문 등록: {qa.question[:50]}... (방식: {qa.method})")
        qa.question= qa.question.lower()
        keyword_str = ""
        question_embedding = None

        if qa.method == "keyword":
            # 키워드 검색 모드: MeCab으로 키워드 추출 후 임베딩
            logger.info("🔍 키워드 모드: MeCab 처리 후 임베딩")

            # 1. 키워드 추출 및 정규화
            keywords = processor.extract_keywords(qa.question)
            normalized_keywords = processor.apply_synonym_replacement(keywords, rep_map)
            keyword_str = " ".join(normalized_keywords)

            logger.info(f"📝 추출된 키워드: {keywords}")
            logger.info(f"🔄 정규화된 키워드: {normalized_keywords}")
            logger.info(f"📄 키워드 문자열: '{keyword_str}'")

            qa.question = keyword_str
            # 2. 키워드 문자열을 임베딩
            if keyword_str.strip():
                question_embedding = embedding_service.get_embedding(keyword_str)
            else:
                logger.warning("⚠️ 추출된 키워드가 없어서 원본 질문으로 임베딩")
                question_embedding = embedding_service.get_embedding(qa.question)
        else:
            # 벡터 검색 모드: 질문 전체를 임베딩
            logger.info("🎯 벡터 모드: 질문 전체 임베딩")
            question_embedding = embedding_service.get_embedding(qa.question)

        if question_embedding is None:
            raise ValueError("임베딩 생성 실패")

        # 3. Redis에 저장
        success = redis_client.store_question_response(
            qa.question,
            qa.answer,
            question_embedding,
        )

        if success:
            logger.info(f"✅ 질문 저장 완료: id={success}")
            return RegisterResponse(
                state=True,
                message="질문 저장 완료",
                id=success
            )
        else:
            raise ValueError("Redis 저장 실패")

    except Exception as e:
        logger.error("❌ 질문 등록 실패", exc_info=e)
        return RegisterResponse(
            state=False,
            message="질문 등록 오류",
            error=str(e)
        )


@app.get("/api/data")
async def get_all_questions(
        limit: int = 10,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "DESC"
):
    """
    전체 질문 목록 조회 API
    """
    try:
        # Redis Search를 사용하여 페이지네이션된 결과 조회
        redis_conn = redis_client._get_redis_connection()

        # 정렬 방향 처리
        sort_direction = "ASC" if sort_order.upper() == "ASC" else "DESC"

        # Redis 스키마에서 use_morphology 필드는 인덱싱하지 않음
        # 단순 저장용 필드로만 사용
        results = redis_conn.execute_command(
            "FT.SEARCH", redis_client.index_name, "*",
            "SORTBY", sort_by, sort_direction,
            "RETURN", 8, "question", "response", "category", "created_at", "last_accessed", "hits", "date_str",
            "use_morphology",
            "LIMIT", offset, limit
        )

        total_count = results[0] if results else 0
        entries = []

        # 결과 파싱
        for i in range(1, len(results), 2):
            if i + 1 < len(results):
                key = results[i]
                data = results[i + 1]

                # 데이터 파싱
                data_iter = iter(data)
                entry_dict = {}
                for k, v in zip(data_iter, data_iter):
                    entry_dict[k] = v

                entry = {
                    "id": key.replace(redis_client.key_prefix, ''),
                    "question": entry_dict.get('question', ''),
                    "response": entry_dict.get('response', ''),
                    "method": entry_dict.get('method', 'vector'),
                    "created_at": int(entry_dict.get('created_at', 0)),
                    "last_accessed": int(entry_dict.get('last_accessed', 0)),
                    "hits": int(entry_dict.get('hits', 0)),
                    "date_str": entry_dict.get('date_str', '')
                }
                entries.append(entry)

        return {
            "state": True,
            "entries": entries,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error("❌ 목록 조회 실패", exc_info=e)
        return {"state": False, "error": str(e)}


@app.delete("/api/data")
async def delete_question(delete_request: Dict[str, str]):
    """
    질문 삭제 API
    """
    try:
        question_id = delete_request.get("id")
        if not question_id:
            raise ValueError("ID가 필요합니다")

        success = redis_client.delete_entry(question_id)

        if success:
            return {"state": True, "message": "삭제 완료"}
        else:
            return {"state": False, "error": "삭제할 항목을 찾을 수 없습니다"}

    except Exception as e:
        logger.error("❌ 삭제 실패", exc_info=e)
        return {"state": False, "error": str(e)}


@app.get("/api/stats")
async def get_cache_stats():
    """
    캐시 통계 API
    """
    try:
        stats = redis_client.get_cache_stats()
        return {"state": True, "stats": stats}
    except Exception as e:
        logger.error("❌ 통계 조회 실패", exc_info=e)
        return {"state": False, "error": str(e)}


# -----------------------------
# ✅ 내부 검색 함수들
# -----------------------------

async def _search_by_vector(query: str) -> SearchResponse:
    """벡터 기반 검색 - 질문 전체를 임베딩"""
    try:
        logger.info(f"🔍 벡터 검색: 질문 전체 임베딩 - '{query}'")

        # 질문 전체를 벡터로 변환
        query_embedding = embedding_service.get_embedding(query)
        if query_embedding is None:
            raise ValueError("임베딩 생성 실패")

        # 벡터 유사도 검색
        result = redis_client.search_similar_question(query_embedding)

        if result["is_cache"] and result["data"]:
            logger.info(f"✅ 벡터 검색 캐시 히트: 유사도 {result['similarity']:.4f}")
            return SearchResponse(
                state=True,
                cached=True,
                similarity=result["similarity"],
                fresh_data={
                    "id": result["data"]["id"],
                    "question": result["data"]["question"],
                    "response": result["data"]["response"],
                    "hits": result["data"]["hits"],
                    "last_accessed": result["data"]["last_accessed"],
                    "created_at": result["data"]["created_at"]
                }
            )
        else:
            logger.info(f"❌ 벡터 검색 캐시 미스: 유사도 {result['similarity']:.4f}")
            return SearchResponse(
                state=True,
                cached=False,
                similarity=result["similarity"]
            )

    except Exception as e:
        logger.error(f"벡터 검색 오류: {e}")
        raise


async def _search_by_keywords(query: str) -> SearchResponse:
    """키워드 기반 검색 - MeCab 처리된 키워드를 임베딩"""
    try:

        logger.info(f"🔍 키워드 검색: MeCab 처리 후 임베딩 - '{query}'")

        # 1. MeCab으로 키워드 추출 및 정규화
        synonym = processor.apply_synonym_replacement(query, rep_map)
        keywords = processor.extract_keywords(query)
        normalized = processor.apply_synonym_replacement(keywords, rep_map)
        keyword_string = " ".join(normalized)

        logger.info(f"📝 원본 : {query}")
        logger.info(f"📝 동의어  : {synonym}")
        logger.info(f"🔄 정규화된 키워드: {normalized}")
        logger.info(f"📄 키워드 문자열: '{keyword_string}'")

        if not keyword_string.strip():
            logger.warning("⚠️ 추출된 키워드가 없습니다.")
            return SearchResponse(
                state=True,
                cached=False,
                similarity=0.0
            )

        # 2. 키워드 문자열을 벡터로 변환
        keyword_embedding = embedding_service.get_embedding(keyword_string)
        if keyword_embedding is None:
            raise ValueError("키워드 임베딩 생성 실패")

        # 3. 벡터 유사도 검색
        result = redis_client.search_similar_question(keyword_embedding)

        if result["is_cache"] and result["data"]:
            logger.info(f"✅ 키워드 검색 캐시 히트: 유사도 {result['similarity']:.4f}")
            return SearchResponse(
                state=True,
                cached=True,
                similarity=result["similarity"],
                fresh_data={
                    "id": result["data"]["id"],
                    "question": result["data"]["question"],
                    "response": result["data"]["response"],
                    "hits": result["data"]["hits"],
                    "last_accessed": result["data"]["last_accessed"],
                    "created_at": result["data"]["created_at"]
                }
            )
        else:
            logger.info(f"❌ 키워드 검색 캐시 미스: 유사도 {result['similarity']:.4f}")
            return SearchResponse(
                state=True,
                cached=False,
                similarity=result["similarity"]
            )

    except Exception as e:
        logger.error(f"키워드 검색 오류: {e}")
        raise


# -----------------------------
# ✅ 레거시 엔드포인트 (기존 호환성)
# -----------------------------

@app.post("/register")
async def legacy_register(qa: QAItem):
    """레거시 등록 엔드포인트 (기존 호환성)"""
    result = await register_question(qa)
    if result.state:
        return {"message": result.message, "id": result.id}
    else:
        raise HTTPException(status_code=500, detail=result.error)





@app.get("/cache_update/")
async def cache_update():
    try:
        synonym_dicts = redis_client.get_redis_json_key("SYNONYM_DICTS")
        print("cache_update============================================================")
        print(synonym_dicts)
        print("cache_update============================================================")
        processor.update_synonym_dict(synonym_dicts)
        return {
            "status": "cache_update",
            "redis": "connected",
            "data": synonym_dicts
        }
    except:
        raise HTTPException(status_code=500)



# -----------------------------
# ✅ 헬스체크
# -----------------------------

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        # Redis 연결 확인
        redis_conn = redis_client._get_redis_connection()
        redis_conn.ping()

        return {
            "status": "healthy",
            "redis": "connected",
            "embedding_service": "ready",
            "mecab_processor": "ready"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)