"""
サンプルプール定義
各カテゴリのFew-shot学習用サンプル
"""

from typing import Dict, List

# カテゴリ別サンプルプール
# サブカテゴリごとに10-20個のサンプルを配分

SAMPLE_POOLS: Dict[str, Dict[str, List[str]]] = {
    "数値計算": {
        "算術演算": [
            "1+2+3+4+5",
            "99-88",
            "12*12",
            "100//3",
            "17%5",
            "2+3*4",
            "10-5+3",
            "8//2",
            "15%4",
            "1+1+1+1+1+1",
            "99*2",
            "144//12",
            "25%7",
            "3*4*5",
            "100-99+1",
        ],
        "べき乗": [
            "2**10",
            "3**5",
            "10**3",
            "5**2",
            "2**8",
            "4**3",
            "7**2",
            "2**16",
            "3**4",
            "10**2",
            "6**2",
            "2**7",
            "5**3",
            "9**2",
            "2**15",
        ],
        "ビット演算": [
            "0xff&0x0f",
            "1<<8",
            "255>>2",
            "~0",
            "0xf|0x0f",
            "0xff^0x0f",
            "1<<16",
            "0xff&0xff",
            "128>>1",
            "0b1111&0b11",
            "1<<10",
            "255>>4",
            "0xff|0x00",
            "0xaa^0x55",
            "1<<7",
        ],
        "複素数": [
            "1+2j",
            "3j",
            "5+0j",
            "0+4j",
            "2+3j",
            "1j",
            "7+1j",
            "9j",
            "4+5j",
            "6+2j",
        ],
        "16進数・8進数": [
            "0xff",
            "0x100",
            "0o777",
            "0xabc",
            "0o17",
            "0xf0",
            "0o100",
            "0xdead",
            "0o77",
            "0x1234",
        ],
    },
    
    "文字列操作": {
        "大文字小文字変換": [
            "'a'.upper()",
            "'AB'.lower()",
            "'hi'.upper()",
            "'Z'.lower()",
            "'test'.upper()",
            "'XYZ'.lower()",
            "'ok'.upper()",
            "'NO'.lower()",
            "'py'.upper()",
            "'GO'.lower()",
        ],
        "文字列繰り返し": [
            "'a'*5",
            "'x'*10",
            "'ab'*3",
            "'*'*16",
            "'!'*8",
            "'='*12",
            "'o'*7",
            "'-'*9",
            "'z'*4",
            "'+'*11",
        ],
        "スライス": [
            "'abcde'[:3]",
            "'xyz'[1:]",
            "'test'[::2]",
            "'hello'[-1]",
            "'py'[::-1]",
            "'ab'[0]",
            "'code'[1:3]",
            "'str'[-2:]",
            "'data'[:2]",
            "'end'[::1]",
        ],
        "split/join": [
            "','.join('ab')",
            "'a b'.split()",
            "' '.join('xy')",
            "''.join('hi')",
            "'a,b'.split(',')",
            "'-'.join('12')",
            "'x'.join('ya')",
            "'a-b'.split('-')",
            "','.join('123')",
            "''.join(['a'])",
        ],
        "インデックスアクセス": [
            "'abc'[0]",
            "'xyz'[1]",
            "'test'[2]",
            "'hello'[-1]",
            "'py'[0]",
            "'data'[-2]",
            "'end'[1]",
            "'str'[0]",
            "'cat'[-1]",
            "'dog'[1]",
        ],
    },
    
    "データ構造": {
        "リスト": [
            "[1,2,3,4,5]",
            "[10,20,30]",
            "list(range(5))",
            "[0]*10",
            "[1,2]+[3,4]",
            "[]",
            "[99,88,77]",
            "[True,False]",
            "['a','b','c']",
            "[1]+[2]+[3]",
            "list('abc')",
            "[0,1,0,1]",
            "[9,8,7,6]",
            "[[1],[2]]",
            "[None,1,2]",
        ],
        "タプル": [
            "(1,2,3)",
            "(10,20)",
            "()",
            "(1,)",
            "1,2,3",
            "(0,0,0)",
            "(True,)",
            "('a','b')",
            "(1,2,3,4)",
            "(99,)",
        ],
        "辞書": [
            "{1:2,3:4}",
            "{'a':1,'b':2}",
            "{0:0}",
            "{}",
            "{1:10}",
            "{'x':99}",
            "{2:4,3:9}",
            "{'k':'v'}",
            "{0:1,1:0}",
            "{5:25}",
        ],
        "セット": [
            "{1,2,3,4}",
            "{10,20,30}",
            "{0}",
            "set()",
            "{1,2,3}",
            "{99,88}",
            "{True}",
            "{1,1,2}",
            "{5,4,3,2,1}",
            "{'a','b'}",
        ],
        "ネスト": [
            "[[1,2],[3]]",
            "[(1,2)]",
            "[{1:2}]",
            "({1:2},)",
            "{1:[2,3]}",
            "[[[]]]",
            "[(),()]",
            "[[1]]",
            "({},)",
            "[{},{}]",
        ],
    },
    
    "内包表記": {
        "リスト内包表記": [
            "[i for i in []]",
            "[x for x in [1]]",
            "[i*2 for i in []]",
            "[0 for _ in [1]]",
            "[i for i in 'ab']",
            "[x+1 for x in []]",
            "[i for i in ()]",
            "[1 for i in [1]]",
            "[c for c in 'xy']",
            "[i**2 for i in []]",
        ],
        "条件付きリスト内包表記": [
            "[i for i in [] if i]",
            "[x for x in [1] if 1]",
            "[i for i in [1,2] if i>1]",
        ],
        "辞書内包表記": [
            "{i:i for i in []}",
            "{x:1 for x in []}",
            "{i:i*2 for i in []}",
            "{c:1 for c in ''}",
            "{i:0 for i in ()}",
        ],
        "セット内包表記": [
            "{i for i in []}",
            "{x for x in ()}",
            "{i*2 for i in []}",
            "{c for c in ''}",
            "{0 for _ in []}",
        ],
        "ジェネレータ式": [
            "(i for i in [])",
            "(x for x in [1])",
            "(i*2 for i in [])",
            "(c for c in 'a')",
            "(0 for _ in [1])",
        ],
        "三項演算子": [
            "1 if True else 0",
            "0 if False else 1",
            "'y' if 1 else 'n'",
            "2 if 1>0 else 3",
            "[] if 0 else [1]",
        ],
    },
    
    "組み込み関数": {
        "集約関数": [
            "len([1,2,3])",
            "sum([1,2,3])",
            "max([1,2,3])",
            "min([5,2,9])",
            "len('hello')",
            "sum([10,20])",
            "max([99,1])",
            "min([3,7])",
            "len([])",
            "sum([])",
            "len((1,2))",
            "max([0,1])",
            "min([9])",
            "sum([0]*5)",
            "len('ab')",
        ],
        "型変換": [
            "int('123')",
            "str(99)",
            "list('abc')",
            "tuple([1,2])",
            "int(1.5)",
            "str(True)",
            "list((1,2))",
            "tuple('xy')",
            "int('0')",
            "str([])",
            "list({})",
            "tuple([])",
            "int(True)",
            "str(None)",
            "list(())",
        ],
        "ソート": [
            "sorted([3,1,2])",
            "sorted('cab')",
            "sorted([9,5,7])",
            "sorted((3,1))",
            "sorted([1])",
            "sorted('zyx')",
            "sorted([0,1,0])",
            "sorted({3,1,2})",
        ],
        "イテレータ": [
            "list(zip([],[]))",
            "list(map(int,[]))",
            "list(filter(None,[]))",
            "list(enumerate([]))",
            "list(reversed([]))",
            "tuple(zip([1],[2]))",
            "list(map(str,[1]))",
        ],
        "述語": [
            "all([1,1,1])",
            "any([0,0,1])",
            "all([])",
            "any([])",
            "all([True])",
            "any([False])",
            "all([1,2])",
            "any([0,1])",
        ],
    },
    
    "標準ライブラリ": {
        "import文": [
            "import os",
            "import sys",
            "import re",
            "import math",
            "import gc",
        ],
        "__import__": [
            "__import__('os')",
            "__import__('sys')",
            "__import__('re')",
            "__import__('gc')",
            "__import__('math')",
        ],
        "複合文": [
            # 16文字制約では難しい
        ],
    },
    
    "その他・応用": {
        "複合演算": [
            "2**10-1",
            "len([1]*10)",
            "sum([1]*5)",
            "max([2**i for i in []])",
            "int('f',16)",
            "str([1,2,3])",
            "sorted([3,1,2])[:2]",
            "[1,2][0]",
            "(1,2,3)[1]",
            "{1:2}[1]",
        ],
        "関数ネスト": [
            "len(str(99))",
            "int(str(1))",
            "str(len('ab'))",
            "max(map(int,['1']))",
            "sum(map(int,[]))",
            "len(list('x'))",
            "str(sum([1,2]))",
            "int(max(['1']))",
        ],
        "複合パターン": [
            "[i**2 for i in [1,2]]",
            "{i:i**2 for i in [1]}",
            "sum([i for i in [1,2]])",
            "[x*2 for x in [1,2,3]]",
            "len([i for i in [1]])",
        ],
    },
}

def get_subcategories(category: str) -> List[str]:
    """
    カテゴリ名からサブカテゴリリストを取得
    
    Args:
        category: カテゴリ名
        
    Returns:
        サブカテゴリ名のリスト
    """
    return list(SAMPLE_POOLS.get(category, {}).keys())


def get_samples(category: str, subcategory: str) -> List[str]:
    """
    カテゴリとサブカテゴリからサンプルリストを取得
    
    Args:
        category: カテゴリ名
        subcategory: サブカテゴリ名
        
    Returns:
        サンプルのリスト
    """
    return SAMPLE_POOLS.get(category, {}).get(subcategory, [])


def get_all_samples(category: str) -> List[str]:
    """
    カテゴリの全サンプルを取得
    
    Args:
        category: カテゴリ名
        
    Returns:
        全サンプルのリスト
    """
    samples = []
    for subcategory in get_subcategories(category):
        samples.extend(get_samples(category, subcategory))
    return samples
