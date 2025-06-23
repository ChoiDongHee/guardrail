# -----------------------------
# 📁 mecabKoreanAnalyzer_service.py (싱글턴 적용 + 로깅 정비 + 주석 보완 + 키워드추출 로직 개선)
# -----------------------------
import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(override=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_stopwords_set(dicts: List[Dict]) -> set:
    """환경 변수에 정의된 불용어 그룹 리스트에서 불용어 단어만 추출하여 집합으로 만듭니다."""
    stopwords = set()
    for group in dicts:
        stopwords.update(group["words"])
    return stopwords


class RefinedKoreanPreprocessor:
    """
    형태소 분석 기반 전처리기 클래스 (싱글턴)
    - 동의어 사전 관리
    - 불용어 처리
    - 형태소 분석 기반 키워드 추출
    """

    _instance = None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, synonym_dict: Optional[List[Dict[str, List[str]]]] = None,
                 stopwords: Optional[List[str]] = None,
                 meaningful_pos: Optional[List[str]] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            import MeCab
            self.tagger = MeCab.Tagger()
            self.logger.info("✅ MeCab 분석기 로딩 완료")
        except ImportError:
            try:
                import mecab_ko as MeCab
                self.tagger = MeCab.Tagger()
                self.logger.info("✅ mecab_ko 대체 로딩 완료")
            except ImportError as e:
                self.logger.exception("❌ MeCab 분석기 로딩 실패")
                raise ImportError("mecab-ko 설치 필요: pip install mecab-ko") from e

        self._synonym_dict = synonym_dict if synonym_dict else []
        self._stopwords = set(stopwords) if stopwords else build_stopwords_set(json.loads(os.getenv("STOPWORDS_DICTS", "[]")))
        self._meaningful_pos = set(meaningful_pos if meaningful_pos else ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'MM', 'MAG', 'SL', 'SN', 'SH'])
        self._initialized = True

    def update_synonym_dict(self, synonym_dicts_json: str) -> bool:
        """JSON 문자열 또는 객체로부터 동의어 사전을 업데이트합니다."""
        try:
            self.logger.info("🔄 동의어 사전 업데이트 시작")
            data = json.loads(synonym_dicts_json) if isinstance(synonym_dicts_json, str) else synonym_dicts_json
            if not isinstance(data, list):
                raise ValueError("동의어 데이터는 리스트 형태여야 합니다")

            for entry in data:
                if not isinstance(entry, dict) or "representative" not in entry or "synonyms" not in entry:
                    raise ValueError("각 항목은 'representative'와 'synonyms' 키 포함 필요")

            self._synonym_dict = data
            self.logger.info(f"✅ 동의어 사전 업데이트 완료 - {len(self._synonym_dict)}개 그룹")
            return True
        except Exception as e:
            self.logger.error(f"❌ 동의어 사전 업데이트 실패: {e}", exc_info=True)
            return False

    def clean_text(self, text: str) -> str:
        """텍스트에서 불필요한 공백을 제거합니다."""
        try:
            return re.sub(r'\s+', ' ', text.strip())
        except Exception as e:
            self.logger.error(f"❌ 텍스트 정제 실패: {e}", exc_info=True)
            return text

    def parse_mecab(self, text: str) -> List[Tuple[str, str]]:
        """
        텍스트를 MeCab 형태소 분석기를 통해 (어간/표제어, 품사) 쌍의 리스트로 반환합니다.
        - lemma: 형태소의 원형 또는 표제어
        - pos: 품사 태그 (예: NNG, NNP 등)
        """
        try:
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
            return morphemes
        except Exception as e:
            self.logger.exception(f"❌ 형태소 분석 실패: {e}")
            return []

    def extract_keywords(self, text: str) -> List[str]:
        """기본 키워드 추출 방식: 형태소 분석 후 불용어 제거 및 동의어 정리"""
        try:
            self.logger.info("🚀 키워드 추출 시작 (기본 방식)")
            return self.extract_keywords_with_synonym(text)
        except Exception as e:
            self.logger.error(f"❌ 키워드 추출 실패: {e}", exc_info=True)
            return []

    def extract_keywords_with_synonym(self, text: str) -> List[str]:
        """
        고급 키워드 추출: 동의어 정리 → 형태소 분석 → 불용어 제거 → 동의어 재적용
        이중 동의어 매핑 구조로 의미 정제를 강화합니다.
        """
        try:
            self.logger.info("🔍 키워드 추출 시작 (동의어-형태소-동의어 방식)")
            rep_map = self.build_synonym_map()
            text_synonym_applied = ' '.join(rep_map.get(word, word) for word in text.split())

            morphemes = self.parse_mecab(self.clean_text(text_synonym_applied))
            keywords = [word for word, pos in morphemes
                        if pos in self._meaningful_pos and word not in self._stopwords and len(word) >= 2 and not re.match(r'^[a-zA-Z]$', word)]

            # 최종 동의어 매핑 적용
            normalized = [rep_map.get(word, word) for word in keywords]
            self.logger.info(f"✅ 키워드 추출 완료: {normalized}")
            return normalized
        except Exception as e:
            self.logger.error(f"❌ 고급 키워드 추출 실패: {e}", exc_info=True)
            return []

    def build_synonym_map(self) -> Dict[str, str]:
        """동의어 사전에서 {동의어: 대표어} 형태의 치환 맵을 생성합니다."""
        try:
            rep_map = {}
            for entry in self._synonym_dict:
                rep = entry.get("representative")
                for word in entry.get("synonyms", []):
                    rep_map[word] = rep
            return rep_map
        except Exception as e:
            self.logger.error(f"❌ 동의어 맵 생성 실패: {e}", exc_info=True)
            return {}

    def apply_synonym_replacement(self, keywords: List[str]) -> List[str]:
        """키워드 리스트에 대해 동의어를 대표어로 치환합니다."""
        try:
            rep_map = self.build_synonym_map()
            return [rep_map.get(word, word) for word in keywords]
        except Exception as e:
            self.logger.error(f"❌ 동의어 정리 실패: {e}", exc_info=True)
            return keywords

    def join_keywords(self, keywords: List[str]) -> str:
        """키워드 리스트를 공백으로 연결된 문자열로 변환합니다."""
        try:
            return ' '.join(keywords)
        except Exception as e:
            self.logger.error(f"❌ 키워드 문자열 변환 실패: {e}", exc_info=True)
            return ''

    # --- Getter / Setter ---

    @property
    def synonym_dict(self):
        return self._synonym_dict

    @synonym_dict.setter
    def synonym_dict(self, new_dict):
        if isinstance(new_dict, list):
            self._synonym_dict = new_dict

    @property
    def stopwords(self):
        return self._stopwords

    @stopwords.setter
    def stopwords(self, new_stopwords):
        if isinstance(new_stopwords, (set, list)):
            self._stopwords = set(new_stopwords)

    @property
    def meaningful_pos(self):
        return self._meaningful_pos

    @meaningful_pos.setter
    def meaningful_pos(self, new_pos):
        if isinstance(new_pos, (set, list)):
            self._meaningful_pos = set(new_pos)
