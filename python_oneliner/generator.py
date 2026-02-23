"""
生成モジュール
LM Studioと通信しワンライナーを生成
"""

import requests
import random
import time
from typing import List, Dict, Optional
from collections import deque

from config import (
    LM_STUDIO_ENDPOINT, LM_STUDIO_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
    BATCH_SIZE, TEMPERATURE_BASE, TEMPERATURE_MID, TEMPERATURE_HIGH,
    MAX_TOKENS, SYSTEM_PROMPT, CATEGORY_ENGLISH, PROMPT_VERBS,
    SAMPLE_HISTORY_SIZE, NEGATIVE_PROMPT_SIZE, TEMPLATE_ROTATION_INTERVAL
)
from sample_pools import get_subcategories, get_samples, get_all_samples


class Generator:
    """生成クラス"""
    
    def __init__(self):
        """初期化"""
        self.batch_counter = 0
        self.category_progress: Dict[str, int] = {}
        
        # バリエーション制御用
        self.sample_history: deque = deque(maxlen=SAMPLE_HISTORY_SIZE)
        self.recent_patterns: deque = deque(maxlen=NEGATIVE_PROMPT_SIZE)
        self.subcategory_history: deque = deque(maxlen=3)
    
    def select_few_shot_samples(self, category: str) -> List[str]:
        """
        Few-shot例を選択
        
        Args:
            category: カテゴリ名
            
        Returns:
            選択されたサンプルのリスト（5個）
        """
        strategy = self.batch_counter % 3
        subcategories = get_subcategories(category)
        
        if strategy == 0:
            # 均等分散戦略
            selected_samples = []
            if len(subcategories) >= 5:
                # 5個のサブカテゴリをランダム選択
                selected_subcats = random.sample(subcategories, 5)
            else:
                # 全サブカテゴリから選択し、不足分は重複選択
                selected_subcats = subcategories * (5 // len(subcategories) + 1)
                selected_subcats = selected_subcats[:5]
            
            # 各サブカテゴリから1個ずつ選択
            for subcat in selected_subcats:
                samples = get_samples(category, subcat)
                # 履歴にないサンプルを優先
                available = [s for s in samples if s not in self.sample_history]
                if available:
                    selected_samples.append(random.choice(available))
                elif samples:
                    selected_samples.append(random.choice(samples))
            
            return selected_samples[:5]
        
        elif strategy == 1:
            # 重点分散戦略
            # 重点サブカテゴリを選択（連続3回同じにならないよう）
            available_subcats = [s for s in subcategories 
                                if s not in self.subcategory_history]
            if not available_subcats:
                available_subcats = subcategories
            
            focus_subcat = random.choice(available_subcats)
            self.subcategory_history.append(focus_subcat)
            
            # 重点サブカテゴリから3個
            focus_samples = get_samples(category, focus_subcat)
            selected_samples = []
            available = [s for s in focus_samples if s not in self.sample_history]
            if len(available) >= 3:
                selected_samples.extend(random.sample(available, 3))
            elif available:
                selected_samples.extend(available)
                remaining = 3 - len(available)
                selected_samples.extend(random.sample(focus_samples, remaining))
            else:
                selected_samples.extend(random.sample(focus_samples, 
                                                      min(3, len(focus_samples))))
            
            # 他のサブカテゴリから各1個
            other_subcats = [s for s in subcategories if s != focus_subcat]
            if other_subcats:
                other_subcats = random.sample(other_subcats, 
                                             min(2, len(other_subcats)))
                for subcat in other_subcats:
                    samples = get_samples(category, subcat)
                    available = [s for s in samples 
                               if s not in self.sample_history]
                    if available:
                        selected_samples.append(random.choice(available))
                    elif samples:
                        selected_samples.append(random.choice(samples))
            
            return selected_samples[:5]
        
        else:
            # ランダム分散戦略
            all_samples = get_all_samples(category)
            available = [s for s in all_samples if s not in self.sample_history]
            if len(available) >= 5:
                return random.sample(available, 5)
            elif available:
                return available + random.sample(all_samples, 5 - len(available))
            else:
                return random.sample(all_samples, min(5, len(all_samples)))
    
    def build_user_prompt(self, category: str, samples: List[str]) -> str:
        """
        ユーザープロンプトを構築
        
        Args:
            category: カテゴリ名
            samples: Few-shot例のリスト
            
        Returns:
            構築されたユーザープロンプト
        """
        category_en = CATEGORY_ENGLISH.get(category, category)
        
        # 戦略に応じたFocus
        strategy = self.batch_counter % 3
        if strategy == 0:
            focus = "balanced across subcategories"
        elif strategy == 1:
            focus = "focused patterns"
        else:
            focus = "mixed patterns"
        
        # 動詞のローテーション
        verb_idx = (self.batch_counter // TEMPLATE_ROTATION_INTERVAL) % len(PROMPT_VERBS)
        verb = PROMPT_VERBS[verb_idx]
        
        # プロンプト構築
        prompt = f"""Category: {category_en}
Focus: {focus}

Examples (16 characters or less):
"""
        
        for i, sample in enumerate(samples, 1):
            prompt += f"{i}. {sample}\n"
        
        prompt += f"""
{verb} {BATCH_SIZE} diverse Python one-liners in this category.
Requirements: 16 characters or less, executable, meaningful.
Vary the patterns and avoid duplicating the examples."""
        
        # ネガティブプロンプト追加
        if self.recent_patterns:
            prompt += "\n\nAvoid generating patterns similar to these recent ones:\n"
            for pattern in list(self.recent_patterns)[:10]:
                prompt += f"- {pattern}\n"
        
        return prompt
    
    def get_temperature(self, category: str, duplicate_rate: float = 0.0) -> float:
        """
        温度パラメータを取得（重複率に応じて動的調整）
        
        Args:
            category: カテゴリ名
            duplicate_rate: 現在の重複率（0.0-1.0）
            
        Returns:
            温度パラメータ
        """
        progress = self.category_progress.get(category, 0)
        
        # 基本温度（進捗に応じて）
        if progress >= 1000:
            base_temp = TEMPERATURE_HIGH
        elif progress >= 500:
            base_temp = TEMPERATURE_MID
        else:
            base_temp = TEMPERATURE_BASE
        
        # 重複率に応じて温度を加算（50%以上で追加）
        if duplicate_rate > 0.5:
            # 50%超えたら+0.05、70%超えたら+0.1
            temp_boost = min(0.15, (duplicate_rate - 0.5) * 0.3)
            return min(1.2, base_temp + temp_boost)  # 最大1.2まで
        
        return base_temp
    
    def generate_batch(self, category: str, duplicate_rate: float = 0.0) -> List[str]:
        """
        LM Studioで1バッチ生成
        
        Args:
            category: カテゴリ名
            duplicate_rate: 現在の重複率（0.0-1.0）
            
        Returns:
            生成されたワンライナーのリスト
        """
        # Few-shot例を選択
        samples = self.select_few_shot_samples(category)
        
        # サンプル履歴に追加
        for sample in samples:
            if sample not in self.sample_history:
                self.sample_history.append(sample)
        
        # ユーザープロンプトを構築
        user_prompt = self.build_user_prompt(category, samples)
        
        # 温度パラメータを取得（重複率を考慮）
        temperature = self.get_temperature(category, duplicate_rate)
        
        # APIリクエストを構築
        payload = {
            "model": "local-model",  # LM Studioのモデル
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
            "n": BATCH_SIZE,
        }
        
        # APIリクエスト送信（リトライ付き）
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    LM_STUDIO_ENDPOINT,
                    json=payload,
                    timeout=LM_STUDIO_TIMEOUT
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # レスポンスからワンライナーを抽出
                    for choice in data.get('choices', []):
                        content = choice.get('message', {}).get('content', '').strip()
                        if content:
                            # 複数行の場合は分割
                            lines = content.split('\n')
                            for line in lines:
                                line = line.strip()
                                # コメント行をスキップ
                                if not line or line.startswith('#'):
                                    continue
                                # 番号や記号を除去
                                if line and not line[0].isdigit():
                                    results.append(line)
                                elif line and '. ' in line:
                                    # 番号付きの場合、番号を除去
                                    code_part = line.split('. ', 1)[1].strip()
                                    # コメント行でないことを確認
                                    if code_part and not code_part.startswith('#'):
                                        results.append(code_part)
                    
                    # バッチカウンタを増加
                    self.batch_counter += 1
                    
                    # カテゴリ進捗を更新
                    self.category_progress[category] = \
                        self.category_progress.get(category, 0) + len(results)
                    
                    return results
                else:
                    print(f"API Error: Status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}/{MAX_RETRIES}")
            except requests.exceptions.ConnectionError:
                print(f"Connection error on attempt {attempt + 1}/{MAX_RETRIES}")
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        
        print("Failed to generate batch after all retries")
        return []
    
    def add_to_recent_patterns(self, pattern: str):
        """
        最近のパターンに追加
        
        Args:
            pattern: パターン文字列
        """
        self.recent_patterns.append(pattern)
