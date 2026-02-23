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

# 目標総数設定
TOTAL_TARGET_DEFAULT = 6000  # デフォルト総数
TOTAL_TARGET_MIN = 600  # 最小総数
TOTAL_TARGET_MAX = 30000  # 最大総数

# カテゴリ名リスト（順序保証）
CATEGORY_NAMES = [
    "数値計算",
    "ビット演算",
    "文字列操作",
    "データ構造",
    "真偽値・条件式",
    "組み込み関数",
    "型変換",
    "スライス記法",
    "標準ライブラリ",
    "その他・応用",
]

# カテゴリ別速度重み（テスト結果から）
# 値が大きいほど速いカテゴリ（多く配分）
CATEGORY_SPEED_WEIGHTS = {
    "数値計算": 1.0,      # 3.20秒/パターン → 普通
    "ビット演算": 2.7,    # 1.19秒 → 速い
    "文字列操作": 3.4,    # 0.95秒 → 速い
    "データ構造": 3.3,     # 0.97秒 → 速い
    "真偽値・条件式": 3.3,  # 0.96秒 → 速い
    "組み込み関数": 0.17,  # 18.40秒 → 非常に遅い
    "型変換": 0.12,     # 27.49秒 → 非常に遅い
    "スライス記法": 4.2,   # 0.76秒 → 速い
    "標準ライブラリ": 0.09,  # 33.66秒 → 非常に遅い
    "その他・応用": 0.23,  # 13.88秒 → 遅い
}

# カテゴリ別最小目標数（品質保証のため）
MIN_CATEGORY_TARGET = 60  # 最小60パターン

def calculate_category_targets(total_target: int = TOTAL_TARGET_DEFAULT) -> List[Tuple[str, int]]:
    """
    総目標数から各カテゴリの目標数を計算
    速度重みに基づいて配分し、遅いカテゴリは少なく、速いカテゴリは多く配分
    
    Args:
        total_target: 総目標パターン数
        
    Returns:
        (category_name, target_count)のリスト
    """
    # 総重みを計算
    total_weight = sum(CATEGORY_SPEED_WEIGHTS[cat] for cat in CATEGORY_NAMES)
    
    # 各カテゴリの目標数を計算
    targets = []
    allocated = 0
    
    for cat in CATEGORY_NAMES:
        weight = CATEGORY_SPEED_WEIGHTS[cat]
        # 重みに応じて配分
        target = int((weight / total_weight) * total_target)
        # 最小値を保証
        target = max(MIN_CATEGORY_TARGET, target)
        targets.append((cat, target))
        allocated += target
    
    # 端数処理：差分を最も速いカテゴリに配分
    diff = total_target - allocated
    if diff != 0:
        # 最も重みが大きい（速い）カテゴリを見つける
        max_weight_idx = max(range(len(CATEGORY_NAMES)), 
                           key=lambda i: CATEGORY_SPEED_WEIGHTS[CATEGORY_NAMES[i]])
        targets[max_weight_idx] = (targets[max_weight_idx][0], 
                                  targets[max_weight_idx][1] + diff)
    
    return targets

# デフォルトのカテゴリ別生成配分（後方互換性のため）
CATEGORIES: List[Tuple[str, int]] = calculate_category_targets(TOTAL_TARGET_DEFAULT)

# 目標総数（後方互換性のため）
TOTAL_TARGET = TOTAL_TARGET_DEFAULT

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
    "数値計算": "Arithmetic Operations",
    "ビット演算": "Bitwise Operations",
    "文字列操作": "String Manipulation",
    "データ構造": "Data Structures",
    "真偽値・条件式": "Boolean and Conditionals",
    "組み込み関数": "Built-in Functions",
    "型変換": "Type Conversions",
    "スライス記法": "Slicing Notation",
    "標準ライブラリ": "Standard Library",
    "その他・応用": "Advanced Patterns",
}

# プロンプト生成用の動詞リスト（ローテーション用）
PROMPT_VERBS = ["Generate", "Create", "Produce"]
