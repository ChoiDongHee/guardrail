import redis
import json
import requests
import time

# ✅ 읽기 쉬운 Python 구조로 정의
synonym_data = [
    {"representative": "파이썬", "synonyms": ["python", "Python", "파이썬", "파이쓴"]},
    {"representative": "웹사이트", "synonyms": ["웹사이트", "웹 사이트", "사이트"]},
    {"representative": "인공지능", "synonyms": ["AI", "에이아이", "인공지능"]},
    {"representative": "데이터", "synonyms": ["데이터", "자료"]},
    {"representative": "모델", "synonyms": ["모형", "모델"]},
    {"representative": "네트워크", "synonyms": ["network", "네트웤","망관리"]},
    {
        "representative": "커피",
        "synonyms": ["커피", "coffee", "카페", "커피음료", "커피메뉴", "커피 한잔", "아메리카노", "라떼", "에스프레소", "아아", "얼죽아","아아아"]
    }
]

# Redis 연결 설정
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_KEY = 'SYNONYM_DICTS'

# API 설정
API_URL = 'http://0.0.0.0:8000/cache_update/'


def save_synonym_dicts():
    """Redis에 동의어 데이터 저장"""
    try:
        # Redis 연결
        r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password='donghee', decode_responses=True)

        # JSON 직렬화
        json_str = json.dumps(synonym_data, ensure_ascii=False)

        # Redis에 저장
        r.set(REDIS_KEY, json_str)
        print(f"✅ Redis 키 '{REDIS_KEY}'에 동의어 정보 저장 완료")

        stored_value = r.get(REDIS_KEY)
        if stored_value:
            print(f"\n📥 저장된 내용 확인:")
            loaded_data = json.loads(stored_value)
            for entry in loaded_data:
                print(f" - 대표어: {entry['representative']}, 동의어: {entry['synonyms']}")
        else:
            print("⚠️ Redis에 저장된 데이터가 없습니다.")
            return False

        return True

    except Exception as e:
        print("❌ Redis 저장 오류:", e)
        return False


def call_cache_update_api():
    """캐시 업데이트 API 호출"""
    try:
        print(f"\n🌐 캐시 업데이트 API 호출 중...")
        print(f"URL: {API_URL}")

        # GET 요청 보내기
        response = requests.get(API_URL, timeout=10)

        print(f"📥 응답 코드: {response.status_code}")

        if response.status_code == 200:
            print("✅ API 호출 성공!")
            try:
                # JSON 응답이면 파싱해서 출력
                json_response = response.json()
                print(f"📄 응답 내용: {json.dumps(json_response, ensure_ascii=False, indent=2)}")
            except:
                # JSON이 아니면 텍스트로 출력
                print(f"📄 응답 내용: {response.text}")
        else:
            print(f"⚠️ API 호출 실패 - 상태 코드: {response.status_code}")
            print(f"📄 응답 내용: {response.text}")
            return False

        return True

    except requests.exceptions.ConnectionError:
        print("❌ API 연결 오류: 서버가 실행 중인지 확인해주세요.")
        return False
    except requests.exceptions.Timeout:
        print("❌ API 요청 시간 초과")
        return False
    except Exception as e:
        print(f"❌ API 호출 오류: {e}")
        return False


def main():
    """메인 실행 함수"""
    print("🚀 Redis 캐시 업데이트 및 API 호출 시작\n")

    # 1단계: Redis에 동의어 데이터 저장
    print("=" * 50)
    print("1단계: Redis 캐시 업데이트")
    print("=" * 50)

    if not save_synonym_dicts():
        print("❌ Redis 업데이트 실패로 인해 작업을 중단합니다.")
        return

    # 잠시 대기
    print(f"\n⏳ API 호출 전 {2}초 대기...")
    time.sleep(2)

    # 2단계: API 호출
    print("\n" + "=" * 50)
    print("2단계: 캐시 업데이트 API 호출")
    print("=" * 50)

    if call_cache_update_api():
        print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️ API 호출에 실패했지만 Redis 업데이트는 완료되었습니다.")


if __name__ == "__main__":
    main()