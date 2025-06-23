
import os

import os


# 가장 먼저, 다른 어떤 임포트보다도 먼저 환경변수 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# 이제 TensorFlow 관련 라이브러리 임포트

from pykospacing import Spacing

spacing = Spacing()

kospacing_result = spacing("아버지가방에들어가신다")

print(kospacing_result)
