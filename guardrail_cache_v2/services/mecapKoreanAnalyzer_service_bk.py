import re
import logging
from typing import List, Dict, Tuple, Optional
import json
import os
from dotenv import load_dotenv
from guardrail_cache_v2.services.redis_service import EnhancedRedisManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)



load_dotenv()

#SYNONYM_DICTS = json.loads(os.getenv('SYNONYM_DICTS'))
STOPWORDS_DICTS = json.loads(os.getenv('STOPWORDS_DICTS'))





def build_stopwords_set(dicts: List[Dict]) -> set:
    stopwords = set()
    for group in dicts:
        stopwords.update(group["words"])
    return stopwords

# -----------------------------
# ✅ 형태소 기반 전처리 클래스
# -----------------------------
class RefinedKoreanPreprocessor:
    def __init__(self,synonym_dict: Optional[List[Dict[str, List[str]]]] = None, stopwords: Optional[List[str]] = None, meaningful_pos: List[str] = None):
        try:
            import MeCab
            self.tagger = MeCab.Tagger()
            logger.info("✅ MeCab 분석기 로딩 완료")
        except ImportError:
            try:
                import mecab_ko as MeCab
                self.tagger = MeCab.Tagger()
                logger.info("✅ mecab_ko 대체 로딩 완료")
            except ImportError as e:
                logger.exception("❌ MeCab 분석기 로딩 실패")
                raise ImportError("mecab-ko 설치 필요: pip install mecab-ko") from e
        self.synonym_dict = synonym_dict if synonym_dict else []
        self.stopwords = set(stopwords) if stopwords else build_stopwords_set(STOPWORDS_DICTS)
        self.meaningful_pos = set(meaningful_pos if meaningful_pos else [
            'NNG', 'NNP', 'NNB', 'VV', 'VA', 'MM', 'MAG', 'SL', 'SN', 'SH'
        ])

    def update_synonym_dict(self, synonym_dicts_json: str) -> bool:
        """
        외부에서 SYNONYM_DICTS JSON을 받아서 self.synonym_dict를 업데이트하는 함수

        Args:
            synonym_dicts_json (str): JSON 문자열 형태의 동의어 사전
                                     예: '[{"representative": "파이썬", "synonyms": ["python", "Python"]}]'

        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            logger.info("🔄 동의어 사전 업데이트 시작")

            # JSON 문자열을 파싱
            if isinstance(synonym_dicts_json, str):
                synonym_data = json.loads(synonym_dicts_json)
            elif isinstance(synonym_dicts_json, list):
                # 이미 파싱된 리스트인 경우
                synonym_data = synonym_dicts_json
            else:
                raise ValueError("입력 데이터는 JSON 문자열 또는 리스트여야 합니다")

            # 데이터 유효성 검증
            if not isinstance(synonym_data, list):
                raise ValueError("동의어 데이터는 리스트 형태여야 합니다")

            for entry in synonym_data:
                if not isinstance(entry, dict):
                    raise ValueError("각 동의어 항목은 딕셔너리 형태여야 합니다")
                if "representative" not in entry or "synonyms" not in entry:
                    raise ValueError("각 항목은 'representative'와 'synonyms' 키를 포함해야 합니다")
                if not isinstance(entry["synonyms"], list):
                    raise ValueError("'synonyms'는 리스트 형태여야 합니다")

            # 기존 동의어 사전 백업 (롤백용)
            old_synonym_dict = self.synonym_dict.copy() if self.synonym_dict else []

            # 새로운 동의어 사전으로 업데이트
            self.synonym_dict = synonym_data

            logger.info(f"✅ 동의어 사전 업데이트 완료 - 총 {len(self.synonym_dict)}개 그룹")

            # 업데이트된 내용 로깅
            for entry in self.synonym_dict:
                logger.debug(f"  - 대표어: {entry['representative']}, 동의어 수: {len(entry['synonyms'])}")

            return True

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 파싱 실패: {e}")
            return False

        except ValueError as e:
            logger.error(f"❌ 데이터 유효성 검증 실패: {e}")
            return False

        except Exception as e:
            logger.error(f"❌ 동의어 사전 업데이트 실패: {e}")
            # 롤백 시도
            try:
                if 'old_synonym_dict' in locals():
                    self.synonym_dict = old_synonym_dict
                    logger.info("🔄 이전 동의어 사전으로 롤백 완료")
            except:
                logger.error("❌ 롤백도 실패했습니다")
            return False



    def clean_text(self, text: str) -> str:
        try:
            return re.sub(r'\s+', ' ', text.strip())
        except Exception as e:
            logger.error(f"❌ 텍스트 정제 실패: {e}")
            return text

    def parse_mecab(self, text: str) -> List[Tuple[str, str]]:
        try:
            logger.debug(f"형태소 분석 시작: {text}")
            result = self.tagger.parse(text)
            morphemes = []
            for line in result.split('\n'):
                if line == 'EOS' or not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    morpheme = parts[0]
                    feature = parts[1].split(',')
                    pos = feature[0]
                    lemma = feature[6] if len(feature) > 6 and feature[6] != '*' else morpheme
                    morphemes.append((lemma, pos))
            logger.debug(f"형태소 분석 결과(표제어): {morphemes}")
            return morphemes
        except Exception as e:
            logger.exception(f"❌ 형태소 분석 실패: {e}")
            return []

    def extract_keywords(self, text: str) -> List[str]:
        try:
            logger.info("🚀 키워드 추출 시작")
            # rep_map = self.build_synonym_map()
            # pre_clean_text = self.clean_text(text)
            #
            # normalized = [rep_map.get(word, word) for word in pre_clean_text]
            #morphemes = self.parse_mecab(normalized)
            morphemes = self.parse_mecab(self.clean_text(text))
            keywords = [
                word for word, pos in morphemes
                if pos in self.meaningful_pos
                and word not in self.stopwords
                and len(word) >= 2
                and not re.match(r'^[a-zA-Z]$', word)
            ]
            logger.info(f"🔍 추출된 키워드: {keywords}")
            # ✅ 대표어 치환 추가
            rep_map = self.build_synonym_map()
            normalized = [rep_map.get(word, word) for word in keywords]

            logger.info(f"🧹 동의어+대표어 치환 추가 정리 결과: {normalized}")
            return normalized

        except Exception as e:
            logger.error(f"❌ 키워드 추출 실패: {e}")
            return []

    def build_synonym_map(self) -> Dict[str, str]:
        rep_map = {}
        try:
            for entry in self.synonym_dict:
                rep = entry["representative"]
                for word in entry["synonyms"]:
                    rep_map[word] = rep
            return rep_map
        except Exception as e:
            logger.error(f"❌ 동의어 맵 생성 실패: {e}")
            return {}

    def apply_synonym_replacement(self, keywords: List[str], rep_map: Dict[str, str]) -> List[str]:
        try:
            normalized = [rep_map.get(word, word) for word in keywords]
            logger.info(f"🧹 동의어 정리 결과: {normalized}")
            return normalized
        except Exception as e:
            logger.error(f"❌ 동의어 정리 실패: {e}")
            return keywords

    def join_keywords(self, keywords: List[str]) -> str:
        try:
            joined = ' '.join(keywords)
            logger.info(f"📝 키워드 스트링 변환: {joined}")
            return joined
        except Exception as e:
            logger.error(f"❌ 키워드 스트링 변환 실패: {e}")
            return ''



