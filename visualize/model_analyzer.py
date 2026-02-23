"""
MicroGPT モデル解析モジュール

学習済みモデルの読み込み、解析、推論機能を提供する。
推論処理はmicrogpt.pyの実装に従い、Python標準ライブラリのみを使用する。
"""

import json
import math
from typing import Dict, List, Tuple, Any, Optional


# ========================================
# 数値演算関数（Python標準ライブラリのみ）
# ========================================

def linear(x: List[float], w: List[List[float]]) -> List[float]:
    """
    行列積を計算：w @ x
    
    Args:
        x: 入力ベクトル, len=n_in
        w: 重み行列, shape=(n_out, n_in)
    
    Returns:
        出力ベクトル, len=n_out
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits: List[float]) -> List[float]:
    """
    数値安定版softmaxを計算
    
    Args:
        logits: 入力ロジット
    
    Returns:
        確率分布（合計が1になるように正規化）
    """
    max_val = max(logits)
    exps = [math.exp(val - max_val) for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x: List[float]) -> List[float]:
    """
    RMS正規化を計算
    
    Args:
        x: 入力ベクトル
    
    Returns:
        正規化されたベクトル
    """
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# ========================================
# ModelAnalyzer クラス
# ========================================

class ModelAnalyzer:
    """学習済みモデルを読み込み、各種解析機能を提供するクラス"""
    
    def __init__(self, model_path: str = '../model.json'):
        """
        初期化処理
        
        Args:
            model_path: モデルファイルのパス（デフォルト: ../model.json）
        
        Raises:
            FileNotFoundError: モデルファイルが見つからない場合
            ValueError: モデルファイルの形式が不正な場合
        """
        try:
            with open(model_path, 'r') as f:
                checkpoint = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"model.jsonが見つかりません（パス: {model_path}）。"
                "プロジェクトルートでmicrogpt.pyを実行してモデルを学習してください。"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"model.jsonの形式が不正です: {e}")
        
        # 必要なキーの存在確認
        required_keys = ['config', 'tokenizer', 'state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(
                f"model.jsonに必要なキーが欠損しています: {', '.join(missing_keys)}"
            )
        
        self.config = checkpoint['config']
        self.tokenizer = checkpoint['tokenizer']
        self.state_dict = checkpoint['state_dict']
        
        # 設定値の取得
        self.vocab_size = self.config['vocab_size']
        self.n_layer = self.config['n_layer']
        self.n_embd = self.config['n_embd']
        self.block_size = self.config['block_size']
        self.n_head = self.config['n_head']
        self.head_dim = self.n_embd // self.n_head
        
        # トークナイザ情報
        self.uchars = self.tokenizer['uchars']
        self.BOS = self.tokenizer['BOS']
        
        # 文字→トークンID辞書を構築
        self.char_to_id = {ch: i for i, ch in enumerate(self.uchars)}
        
        print(f"モデル読み込み完了: vocab_size={self.vocab_size}, "
              f"n_layer={self.n_layer}, n_embd={self.n_embd}, "
              f"block_size={self.block_size}, n_head={self.n_head}")
    
    def get_parameter_stats(self) -> Dict[str, Any]:
        """
        各レイヤーのパラメータ数と統計量を計算
        
        Returns:
            統計情報の辞書
            - 'params_per_layer': {layer_name: param_count}
            - 'stats_per_layer': {layer_name: {'mean': float, 'std': float, 'min': float, 'max': float}}
            - 'total_params': int
        """
        import numpy as np
        
        params_per_layer = {}
        stats_per_layer = {}
        total_params = 0
        
        for layer_name, matrix in self.state_dict.items():
            # パラメータ数を計算
            n_params = len(matrix) * len(matrix[0])
            params_per_layer[layer_name] = n_params
            total_params += n_params
            
            # numpy配列に変換して統計量を計算
            arr = np.array(matrix)
            stats_per_layer[layer_name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }
        
        return {
            'params_per_layer': params_per_layer,
            'stats_per_layer': stats_per_layer,
            'total_params': total_params
        }
    
    def get_embedding_matrix(self, embedding_type: str = 'token'):
        """
        埋め込み行列をnumpy配列として返す
        
        Args:
            embedding_type: "token" または "position"
        
        Returns:
            numpy配列
            - token: shape=(vocab_size, n_embd)
            - position: shape=(block_size, n_embd)
        
        Raises:
            ValueError: 不正なembedding_typeが指定された場合
        """
        import numpy as np
        
        if embedding_type == 'token':
            return np.array(self.state_dict['wte'])
        elif embedding_type == 'position':
            return np.array(self.state_dict['wpe'])
        else:
            raise ValueError(f"embedding_typeは'token'または'position'を指定してください: {embedding_type}")
    
    def get_attention_weights(self, layer_idx: int) -> Dict[str, Any]:
        """
        指定レイヤーのAttention重み行列を取得
        
        Args:
            layer_idx: レイヤーインデックス（0-indexed）
        
        Returns:
            辞書形式 {"wq": array, "wk": array, "wv": array, "wo": array}
        
        Raises:
            ValueError: 不正なlayer_idxが指定された場合
        """
        import numpy as np
        
        if not (0 <= layer_idx < self.n_layer):
            raise ValueError(f"layer_idxは0から{self.n_layer-1}の範囲で指定してください: {layer_idx}")
        
        return {
            'wq': np.array(self.state_dict[f'layer{layer_idx}.attn_wq']),
            'wk': np.array(self.state_dict[f'layer{layer_idx}.attn_wk']),
            'wv': np.array(self.state_dict[f'layer{layer_idx}.attn_wv']),
            'wo': np.array(self.state_dict[f'layer{layer_idx}.attn_wo']),
        }
    
    def get_mlp_weights(self, layer_idx: int) -> Dict[str, Any]:
        """
        指定レイヤーのMLP重み行列を取得
        
        Args:
            layer_idx: レイヤーインデックス（0-indexed）
        
        Returns:
            辞書形式 {"fc1": array, "fc2": array}
        
        Raises:
            ValueError: 不正なlayer_idxが指定された場合
        """
        import numpy as np
        
        if not (0 <= layer_idx < self.n_layer):
            raise ValueError(f"layer_idxは0から{self.n_layer-1}の範囲で指定してください: {layer_idx}")
        
        return {
            'fc1': np.array(self.state_dict[f'layer{layer_idx}.mlp_fc1']),
            'fc2': np.array(self.state_dict[f'layer{layer_idx}.mlp_fc2']),
        }
    
    def tokenize(self, text: str) -> List[int]:
        """
        文字列をトークンID列に変換
        
        Args:
            text: 入力文字列
        
        Returns:
            トークンIDのリスト（BOSは含まない）
        """
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        トークンID列を文字列に変換
        
        Args:
            token_ids: トークンIDのリスト
        
        Returns:
            文字列（BOSは無視）
        """
        chars = []
        for token_id in token_ids:
            if token_id == self.BOS:
                continue
            if 0 <= token_id < len(self.uchars):
                chars.append(self.uchars[token_id])
        return ''.join(chars)
    
    def get_phonetic_category(self, char: str) -> str:
        """
        文字の音韻的カテゴリまたはPython記号の詳細カテゴリを取得
        
        Args:
            char: 文字（'a', 'b', 'c', ...）または記号
        
        Returns:
            カテゴリ名（11カテゴリ）
        """
        # 音韻的特徴（英字用）
        if char in 'aeiouAEIOU':
            return 'vowels'
        elif char in 'pbtdkgPBTDKG':
            return 'plosives'
        elif char in 'fvszhxFVSZHX':
            return 'fricatives'
        elif char in 'mnlrMNLR':
            return 'sonorants'
        elif char in 'wyWY':
            return 'approximants'
        # Python記号の詳細分類
        elif char in '()':
            return 'round_brackets'
        elif char in '[]':
            return 'square_brackets'
        elif char in '{}':
            return 'curly_brackets'
        elif char in '+-*/%=<>&|^~!@':
            return 'operators'
        elif char in ',:;.':
            return 'delimiters'
        elif char in '\'"`':
            return 'quotes'
        else:
            return 'whitespace_other'
    
    def get_phonetic_categories_dict(self) -> Dict[str, str]:
        """
        全トークンに対する音韻的カテゴリ＋Python記号の詳細カテゴリの辞書を返す
        
        Returns:
            {token_char: category_name} の辞書
        """
        return {ch: self.get_phonetic_category(ch) for ch in self.uchars}


# ========================================
# InferenceEngine クラス
# ========================================

class InferenceEngine:
    """学習済みモデルを使って推論を実行し、中間状態をキャプチャするクラス"""
    
    def __init__(self, analyzer: ModelAnalyzer):
        """
        初期化処理
        
        Args:
            analyzer: ModelAnalyzerインスタンス
        """
        self.analyzer = analyzer
        self.state_dict = analyzer.state_dict
        
        # 設定値をコピー
        self.vocab_size = analyzer.vocab_size
        self.n_layer = analyzer.n_layer
        self.n_embd = analyzer.n_embd
        self.block_size = analyzer.block_size
        self.n_head = analyzer.n_head
        self.head_dim = analyzer.head_dim
        self.BOS = analyzer.BOS
    
    def _gpt_forward(self, token_id: int, pos_id: int, keys: List[List], values: List[List],
                     record_states: bool = False) -> Tuple[List[float], Optional[Dict]]:
        """
        GPTモデルの順伝播（microgpt.pyの実装と同じ）
        
        Args:
            token_id: 入力トークンID
            pos_id: 位置ID
            keys: 各レイヤーのキーキャッシュ
            values: 各レイヤーのバリューキャッシュ
            record_states: 中間状態を記録するかどうか
        
        Returns:
            logits: 次トークンのロジット
            states: 中間状態の辞書（record_states=Trueの場合のみ）
        """
        states = {} if record_states else None
        
        # Embedding
        tok_emb = self.state_dict['wte'][token_id]
        pos_emb = self.state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)
        
        if record_states:
            states['embedding'] = x.copy()
        
        # Transformer layers
        for li in range(self.n_layer):
            # 1) Multi-head Attention block
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, self.state_dict[f'layer{li}.attn_wq'])
            k = linear(x, self.state_dict[f'layer{li}.attn_wk'])
            v = linear(x, self.state_dict[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)
            
            x_attn = []
            attn_weights_all_heads = []
            
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5 
                              for t in range(len(k_h))]
                attn_weights = softmax(attn_logits)
                attn_weights_all_heads.append(attn_weights)
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                           for j in range(self.head_dim)]
                x_attn.extend(head_out)
            
            if record_states:
                states[f'layer{li}_attn_weights'] = attn_weights_all_heads
            
            x = linear(x_attn, self.state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]
            
            if record_states:
                states[f'layer{li}_attn_out'] = x.copy()
            
            # 2) MLP block
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, self.state_dict[f'layer{li}.mlp_fc1'])
            x = [max(0, xi) for xi in x]  # ReLU
            x = linear(x, self.state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]
            
            if record_states:
                states[f'layer{li}_mlp_out'] = x.copy()
        
        # LM head
        logits = linear(x, self.state_dict['lm_head'])
        
        if record_states:
            states['logits'] = logits.copy()
        
        return logits, states
    
    def run_inference(self, input_text: str, max_length: int = 10, 
                     temperature: float = 0.5, record_states: bool = True) -> Tuple[str, Optional[Dict]]:
        """
        自己回帰的に推論を実行
        
        Args:
            input_text: 入力テキスト（空文字列の場合はBOSから開始）
            max_length: 最大生成トークン数
            temperature: サンプリング温度（0に近いほど確定的）
            record_states: 中間状態を記録するかどうか
        
        Returns:
            generated_text: 生成されたテキスト
            all_states: 中間状態の辞書（record_states=Trueの場合のみ）
        """
        import random
        random.seed(42)
        
        # トークン化
        if input_text:
            tokens = [self.BOS] + self.analyzer.tokenize(input_text)
        else:
            tokens = [self.BOS]
        
        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]
        all_states = [] if record_states else None
        
        # 自己回帰生成
        for step in range(max_length):
            pos_id = len(tokens) - 1
            if pos_id >= self.block_size:
                break
            
            token_id = tokens[-1]
            logits, states = self._gpt_forward(token_id, pos_id, keys, values, record_states)
            
            # サンプリング
            probs = softmax([l / temperature for l in logits])
            next_token_id = random.choices(range(self.vocab_size), weights=probs)[0]
            
            if record_states:
                states['probs'] = probs
                states['next_token_id'] = next_token_id
                states['token_id'] = token_id
                states['pos_id'] = pos_id
                all_states.append(states)
            
            if next_token_id == self.BOS:
                break
            
            tokens.append(next_token_id)
        
        # 生成されたテキスト（BOSを除く）
        generated_text = self.analyzer.detokenize(tokens[1:])
        
        return generated_text, all_states
    
    def compute_attention_for_sequence(self, token_ids: List[int]):
        """
        指定されたトークン列に対してAttention重みを計算
        
        Args:
            token_ids: トークンIDのリスト（BOSを含む）
        
        Returns:
            numpy配列, shape=(n_layer, n_head, seq_len, seq_len)
        """
        import numpy as np
        
        seq_len = len(token_ids)
        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]
        
        # 全位置のAttention重みを収集
        all_attn_weights = np.zeros((self.n_layer, self.n_head, seq_len, seq_len))
        
        for pos_id, token_id in enumerate(token_ids):
            _, states = self._gpt_forward(token_id, pos_id, keys, values, record_states=True)
            
            for li in range(self.n_layer):
                attn_weights_all_heads = states[f'layer{li}_attn_weights']
                for h in range(self.n_head):
                    attn_weights = attn_weights_all_heads[h]
                    # 現在の位置から各過去位置へのAttention重み
                    for t in range(len(attn_weights)):
                        all_attn_weights[li, h, pos_id, t] = attn_weights[t]
        
        return all_attn_weights
    
    def get_token_probabilities(self, context_token_ids: List[int]):
        """
        指定されたコンテキストに対して次トークンの確率分布を計算
        
        Args:
            context_token_ids: コンテキストのトークンIDリスト（BOSを含む）
        
        Returns:
            numpy配列, shape=(vocab_size,)
        """
        import numpy as np
        
        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]
        
        # コンテキストを順に処理
        for pos_id, token_id in enumerate(context_token_ids):
            logits, _ = self._gpt_forward(token_id, pos_id, keys, values, record_states=False)
        
        # 最後の位置のlogitsからsoftmax
        probs = softmax(logits)
        
        return np.array(probs)
