"""
設定ファイル
16文字Pythonワンライナー生成システムの設定
"""

from typing import Dict, List, Tuple

# LM Studio API設定
LM_STUDIO_ENDPOINT = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_TIMEOUT = 30  # 秒
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

# 生成設定
BATCH_SIZE = 10  # 一度に生成するパターン数
TEMPERATURE_BASE = 0.8
TEMPERATURE_MID = 0.85  # 500パターン生成後
TEMPERATURE_HIGH = 0.9  # 1000パターン生成後
MAX_TOKENS = 100

# 検証設定
MAX_LENGTH = 16  # 最大文字数
MIN_LENGTH = 1   # 最小文字数
EXECUTION_TIMEOUT = 1  # 秒

# 重複排除設定
LEVENSHTEIN_THRESHOLD = 5  # 類似判定の編集距離閾値（緩和して多様性を確保）

# バリエーション制御設定
SAMPLE_HISTORY_SIZE = 3  # バッチ
NEGATIVE_PROMPT_SIZE = 20  # パターン
TEMPLATE_ROTATION_INTERVAL = 100  # バッチ

# カテゴリ別生成配分
CATEGORIES: List[Tuple[str, int]] = [
    ("数値計算", 7000),
    ("文字列操作", 7000),
    ("データ構造", 7000),
    ("組み込み関数", 7000),
    ("標準ライブラリ", 1500),
    ("その他・応用", 500),
]

# 目標総数
TOTAL_TARGET = 30000

# 出力ファイル名
OUTPUT_FILENAME = "python_oneliners_30k.txt"

# システムプロンプト（英語）
SYSTEM_PROMPT = """You are a Python expert. Generate meaningful Python one-liners with 16 characters or fewer.

Requirements:
- 16 characters or less (1-16 characters)
- Executable in Python REPL
- No syntax errors
- Returns a value or has side effects
- Creative and diverse patterns
- Output ONLY executable code, NO comments, NO explanations

Avoid:
- Code that causes errors
- Meaningless character strings
- Duplicate or similar patterns
- Comments (lines starting with #)
- Incomplete code"""

# カテゴリ別英語名マッピング
CATEGORY_ENGLISH: Dict[str, str] = {
    "数値計算": "Numeric Computation",
    "文字列操作": "String Manipulation",
    "データ構造": "Data Structures",
    "組み込み関数": "Built-in Functions",
    "標準ライブラリ": "Standard Library",
    "その他・応用": "Advanced Patterns",
}

# プロンプト生成用の動詞リスト（ローテーション用）
PROMPT_VERBS = ["Generate", "Create", "Produce"]
