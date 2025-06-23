import pandas as pd
import re
import time
from guardrail_cache_v2.services.redis_service import QuestionAnswerCacheService
from guardrail_cache_v2.services.mecapKoreanAnalyzer_service import RefinedKoreanPreprocessor
from guardrail_cache_v2.services.redis_service import EnhancedRedisManager, EmbeddingService


def extract_qa_from_block(text: str):
    """
    하나의 블록에서 '질문:'과 '답변:'을 추출하는 함수
    """
    if not isinstance(text, str):
        return None, None

    # 큰 따옴표 제거 및 양쪽 공백 제거
    text = text.strip().strip('"').strip()

    # 줄바꿈 통합 후 추출
    pattern = r"질문:\s*(.*?)\s*답변:\s*(.*)"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        return question, answer
    return None, None


def bulk_insert_from_blocks(csv_path: str, similarity_threshold: float = 0.94):
    df = pd.read_csv(csv_path, header=None)  # 헤더 없음으로 가정
    print(f"\n📥 [파일 읽기 완료] '{csv_path}'에서 {len(df)}개의 Q&A 추출 시도\n")

    success, skipped, fail = 0, 0, 0

    for idx, row in df.iterrows():
        text = row[0]
        question, answer = extract_qa_from_block(text)

        if not question or not answer:
            print(f"⚠️ [{idx+1}] Q/A 파싱 실패 → {text[:50]}...")
            fail += 1
            continue

        # 1. 키워드 전처리 및 정규화
        keywords = processor.extract_keywords(question)
        normalized_keywords = processor.apply_synonym_replacement(keywords, rep_map)
        keyword_str = " ".join(normalized_keywords).lower()

        if not keyword_str.strip():
            print(f"⚠️ [{idx+1}] 키워드 없음 → '{question[:30]}...' → 생략")
            fail += 1
            continue

        print(f"\n[{idx+1}] 📄 질문 키워드: '{keyword_str}'")

        # 2. 임베딩 생성
        embedding = embedding_service.get_embedding(keyword_str)
        if embedding is None:
            print(f"❌ [{idx+1}] 임베딩 생성 실패")
            fail += 1
            continue

        # 3. 유사도 검사
        result = redis_client.search_similar_question(embedding, limit=1)
        similarity = result.get("similarity", 0.0)
        print(f"🔍 유사도 검사: {similarity:.4f}")

        if similarity >= similarity_threshold:
            print(f"⛔ 유사도 {similarity:.4f} ≥ {similarity_threshold} → 저장 생략")
            skipped += 1
            continue

        # 4. 저장
        redis_client.store_question_response(keyword_str, answer, embedding)
        print(f"✅ 저장 완료")
        success += 1

        time.sleep(0.01)

    print("\n📊 [요약 결과]")
    print(f"   - 저장 성공: {success}")
    print(f"   - 유사도 초과로 건너뜀: {skipped}")
    print(f"   - 실패 (파싱/임베딩 등): {fail}")


if __name__ == "__main__":
    # -----------------------------
    # ✅ 서비스 초기화
    # -----------------------------
    redis_client = EnhancedRedisManager()
    processor = RefinedKoreanPreprocessor()
    rep_map = processor.build_synonym_map()
    embedding_service = EmbeddingService.get_instance()

    # -----------------------------
    # 🔄 인서트 실행
    # -----------------------------
    bulk_insert_from_blocks("퇴직연금_자주묻는질문_QA.csv")
    bulk_insert_from_blocks("퇴직연금_시나리오_200.csv")
