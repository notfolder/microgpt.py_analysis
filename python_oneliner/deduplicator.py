"""
重複排除モジュール
完全一致および類似パターンの検出・除外
"""

from typing import Set, Dict, List
from config import LEVENSHTEIN_THRESHOLD


class Deduplicator:
    """重複排除クラス"""
    
    def __init__(self):
        """初期化"""
        # 完全一致検出用set
        self.existing_set: Set[str] = set()
        
        # 類似パターン検出用（文字数グループ化 + ハッシュテーブル）
        self.length_groups: Dict[int, Dict[str, List[str]]] = {}
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Levenshtein距離（編集距離）を計算
        
        Args:
            s1: 文字列1
            s2: 文字列2
            
        Returns:
            編集距離
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 挿入、削除、置換のコストを計算
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def is_exact_duplicate(self, code: str) -> bool:
        """
        完全一致重複チェック
        
        Args:
            code: チェックするコード文字列
            
        Returns:
            重複している場合True
        """
        return code in self.existing_set
    
    def is_similar_duplicate(self, code: str) -> bool:
        """
        類似パターン重複チェック
        
        Args:
            code: チェックするコード文字列
            
        Returns:
            類似パターンが存在する場合True
        """
        length = len(code)
        
        # 文字数グループが存在しない場合は重複なし
        if length not in self.length_groups:
            return False
        
        # ハッシュキー（先頭3文字）を計算
        hash_key = code[:3] if len(code) >= 3 else code
        
        # 同一ハッシュキーのパターンが存在しない場合は重複なし
        if hash_key not in self.length_groups[length]:
            return False
        
        # 同一ハッシュキー内でLevenshtein距離を計算
        for existing_code in self.length_groups[length][hash_key]:
            distance = self.levenshtein_distance(code, existing_code)
            if distance < LEVENSHTEIN_THRESHOLD:
                return True
        
        return False
    
    def is_duplicate(self, code: str) -> bool:
        """
        重複チェック（完全一致 + 類似パターン）
        
        Args:
            code: チェックするコード文字列
            
        Returns:
            重複している場合True
        """
        # 完全一致チェック
        if self.is_exact_duplicate(code):
            return True
        
        # 類似パターンチェック
        if self.is_similar_duplicate(code):
            return True
        
        return False
    
    def add(self, code: str):
        """
        コードを重複検出データ構造に追加
        
        Args:
            code: 追加するコード文字列
        """
        # 完全一致検出用setに追加
        self.existing_set.add(code)
        
        # 類似パターン検出用に追加
        length = len(code)
        
        # 文字数グループを作成（存在しない場合）
        if length not in self.length_groups:
            self.length_groups[length] = {}
        
        # ハッシュキー（先頭3文字）を計算
        hash_key = code[:3] if len(code) >= 3 else code
        
        # ハッシュテーブルに追加
        if hash_key not in self.length_groups[length]:
            self.length_groups[length][hash_key] = []
        
        self.length_groups[length][hash_key].append(code)
    
    def count(self) -> int:
        """
        登録済みパターン数を取得
        
        Returns:
            登録済みパターン数
        """
        return len(self.existing_set)
