"""
検証モジュール
生成されたワンライナーの妥当性を検証
"""

import signal
from typing import Dict, Any
from contextlib import contextmanager
from config import MAX_LENGTH, MIN_LENGTH, EXECUTION_TIMEOUT


class TimeoutException(Exception):
    """タイムアウト例外"""
    pass


@contextmanager
def time_limit(seconds: int):
    """
    実行時間制限のコンテキストマネージャ
    
    Args:
        seconds: タイムアウト秒数
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def validate_length(code: str) -> bool:
    """
    文字数検証
    
    Args:
        code: 検証するコード文字列
        
    Returns:
        検証結果（True: 通過, False: 不合格）
    """
    length = len(code)
    return MIN_LENGTH <= length <= MAX_LENGTH


def validate_syntax(code: str) -> bool:
    """
    構文検証
    
    Args:
        code: 検証するコード文字列
        
    Returns:
        検証結果（True: 通過, False: 不合格）
    """
    # まずeval modeでコンパイルを試行
    try:
        compile(code, '<string>', 'eval')
        return True
    except SyntaxError:
        pass
    
    # eval modeで失敗した場合はexec modeで試行
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def validate_execution(code: str) -> bool:
    """
    実行検証
    
    Args:
        code: 検証するコード文字列
        
    Returns:
        検証結果（True: 通過, False: 不合格）
    """
    # サンドボックス環境の設定
    safe_globals = {'__builtins__': {}}
    safe_locals = {}
    
    try:
        with time_limit(EXECUTION_TIMEOUT):
            # まずeval modeで実行試行
            try:
                eval(code, safe_globals, safe_locals)
                return True
            except:
                pass
            
            # eval modeで失敗した場合はexec modeで試行
            try:
                exec(code, safe_globals, safe_locals)
                return True
            except:
                return False
    except TimeoutException:
        return False
    except Exception:
        return False


def validate(code: str) -> Dict[str, Any]:
    """
    ワンライナーの総合検証
    
    Args:
        code: 検証するコード文字列
        
    Returns:
        検証結果の辞書
        - valid: 全検証を通過したか
        - length: 文字数
        - length_ok: 文字数検証結果
        - syntax_ok: 構文検証結果
        - executable: 実行検証結果
        - error: エラーメッセージ（エラー時のみ）
    """
    result = {
        'valid': False,
        'length': len(code),
        'length_ok': False,
        'syntax_ok': False,
        'executable': False,
    }
    
    # コメント行チェック
    code_stripped = code.strip()
    if not code_stripped or code_stripped.startswith('#'):
        result['error'] = "Comment or empty line"
        return result
    
    # 文字数検証
    result['length_ok'] = validate_length(code)
    if not result['length_ok']:
        result['error'] = f"Length error: {len(code)} characters"
        return result
    
    # 構文検証
    result['syntax_ok'] = validate_syntax(code)
    if not result['syntax_ok']:
        result['error'] = "Syntax error"
        return result
    
    # 実行検証
    result['executable'] = validate_execution(code)
    if not result['executable']:
        result['error'] = "Execution error"
        return result
    
    # 全検証通過
    result['valid'] = True
    return result
