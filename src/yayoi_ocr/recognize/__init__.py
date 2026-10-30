"""認識エンジン。

- `types` … 認識結果の型とインターフェース
- `parseq` … 汎用の行認識 R1(NDLOCR-Lite の PARSeq tiny を取り込んだもの)
"""

from .types import Recognition, Recognizer

__all__ = ["Recognition", "Recognizer"]
