import redis

r = redis.Redis(host='localhost', port=6379, db=0, password='donghee')


def safe_delete_keys(pattern):
    """삭제 전 확인"""
    # 먼저 키들 확인
    keys = list(r.scan_iter(match=pattern))

    if not keys:
        print("삭제할 키가 없습니다.")
        return 0

    print(f"삭제될 키 목록 ({len(keys)}개):")
    for key in keys[:10]:  # 처음 10개만 표시
        print(f"  - {key.decode('utf-8')}")

    if len(keys) > 10:
        print(f"  ... 그리고 {len(keys) - 10}개 더")

    # 확인
    confirm = input(f"\n정말로 {len(keys)}개의 키를 삭제하시겠습니까? (y/N): ")

    if confirm.lower() == 'y':
        # 삭제 실행
        if keys:
            r.delete(*keys)
            print(f"{len(keys)}개의 키가 삭제되었습니다.")
            return len(keys)
    else:
        print("삭제가 취소되었습니다.")
        return 0


# 사용
safe_delete_keys("service:*")