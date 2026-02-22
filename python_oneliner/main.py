"""
メインプログラム
16文字Pythonワンライナー生成システム
"""

import time
import argparse
from typing import Dict
from datetime import timedelta

from config import CATEGORIES, OUTPUT_FILENAME, TOTAL_TARGET
from generator import Generator
from validator import validate
from deduplicator import Deduplicator


def main():
    """メイン処理"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description='16文字Pythonワンライナー生成システム'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=OUTPUT_FILENAME,
        help=f'出力ファイル名 (default: {OUTPUT_FILENAME})'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='テストモード（各カテゴリ10パターンのみ生成）'
    )
    args = parser.parse_args()
    
    # テストモードの場合は目標数を調整
    if args.test:
        categories = [(name, 10) for name, _ in CATEGORIES]
        print("=== テストモード ===")
    else:
        categories = CATEGORIES
    
    # 開始時刻
    start_time = time.time()
    
    # モジュール初期化
    generator = Generator()
    deduplicator = Deduplicator()
    
    # カウンタ初期化
    total_generated = 0
    validated_count = 0
    duplicate_count = 0
    
    # カテゴリ別達成数
    category_counts: Dict[str, int] = {name: 0 for name, _ in categories}
    
    # カテゴリ別所要時間
    category_times: Dict[str, float] = {}
    
    # 出力ファイルオープン
    print(f"出力ファイル: {args.output}")
    
    # 本番のカテゴリ別目標数（推定用）
    production_targets = dict(CATEGORIES)
    total_categories = len(categories)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        # カテゴリループ
        for category_idx, (category_name, target_count) in enumerate(categories, 1):
            print(f"\n{'='*60}")
            print(f"カテゴリ: {category_name} ({category_idx}/{total_categories})")
            print(f"目標: {target_count}パターン")
            print('='*60)
            
            # カテゴリ開始時刻を記録
            category_start_time = time.time()
            
            category_count = 0
            consecutive_errors = 0
            
            # カテゴリ目標数に達するまでループ
            while category_count < target_count:
                # バッチ生成
                try:
                    batch_results = generator.generate_batch(category_name)
                    consecutive_errors = 0
                except Exception as e:
                    print(f"生成エラー: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        print("連続エラーが10回に達しました。プログラムを終了します。")
                        return
                    continue
                
                if not batch_results:
                    print("バッチ生成に失敗しました")
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        print("連続エラーが10回に達しました。プログラムを終了します。")
                        return
                    continue
                
                # バッチ内の各パターンを処理
                for code in batch_results:
                    total_generated += 1
                    
                    # 文字数検証
                    if len(code) == 0:
                        continue
                    
                    # 検証
                    validation_result = validate(code)
                    
                    if not validation_result['valid']:
                        # 検証失敗
                        continue
                    
                    # 重複チェック
                    if deduplicator.is_duplicate(code):
                        duplicate_count += 1
                        continue
                    
                    # 検証通過 & 重複なし -> ファイルに書き込み
                    f.write(code + '\n')
                    f.flush()
                    
                    validated_count += 1
                    category_count += 1
                    category_counts[category_name] += 1
                    
                    # 重複排除データ構造に追加
                    deduplicator.add(code)
                    
                    # 最近のパターンに追加
                    generator.add_to_recent_patterns(code)
                    
                    # カテゴリ目標達成チェック
                    if category_count >= target_count:
                        break
                
                # 進捗表示（100パターンごと）
                if total_generated % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"進捗: {total_generated} 生成 | "
                          f"{validated_count} 検証通過 | "
                          f"{duplicate_count} 重複除外 | "
                          f"経過時間: {timedelta(seconds=int(elapsed))}")
            
            # カテゴリ完了
            category_end_time = time.time()
            category_elapsed = category_end_time - category_start_time
            category_times[category_name] = category_elapsed
            print(f"\n{category_name} 完了: {category_count}/{target_count} パターン")
            print(f"所要時間: {timedelta(seconds=int(category_elapsed))}")
            
            # テストモードの場合、即座に推定値を表示
            if args.test and category_count > 0:
                production_target = production_targets[category_name]
                time_per_pattern = category_elapsed / category_count
                estimated_time = time_per_pattern * production_target
                estimated_timedelta = timedelta(seconds=int(estimated_time))
                estimated_hours = estimated_time / 3600
                
                print(f"\n📈 本番実行推定:")
                print(f"  - 目標パターン数: {production_target:,}")
                print(f"  - 1パターンあたり: {time_per_pattern:.2f}秒")
                print(f"  - 推定所要時間: {estimated_timedelta} ({estimated_hours:.2f}時間)")
    
    # 終了処理
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("=== 最終統計 ===")
    print('='*60)
    print(f"総生成数: {total_generated}")
    print(f"検証通過数: {validated_count}")
    print(f"重複除外数: {duplicate_count}")
    print(f"処理時間: {timedelta(seconds=int(elapsed_time))}")
    
    print(f"\n{'='*60}")
    print("=== カテゴリ別達成数 ===")
    print('='*60)
    for category_name, target_count in categories:
        achieved = category_counts[category_name]
        percentage = (achieved / target_count * 100) if target_count > 0 else 0
        print(f"{category_name:12s}: {achieved:5d}/{target_count:5d} ({percentage:6.2f}%)")
    
    # 成功基準の判定
    total_target = sum(target for _, target in categories)
    if validated_count >= total_target:
        print(f"\n✓ 成功: 目標{total_target}パターンを達成しました！")
    else:
        shortage = total_target - validated_count
        print(f"\n✗ 不足: 目標まであと{shortage}パターン必要です")
    
    # テストモードの場合、カテゴリごとの推定時間を表示
    if args.test and validated_count > 0:
        print(f"\n{'='*60}")
        print("=== カテゴリ別 所要時間推定 ===")
        print('='*60)
        
        # 本番のカテゴリ別目標数
        production_targets = dict(CATEGORIES)
        
        total_estimated_time = 0
        
        print(f"\n{'カテゴリ':12s} | {'テスト':>8s} | {'本番目標':>8s} | {'所要時間':>12s} | {'推定時間':>12s}")
        print("-" * 70)
        
        for category_name, test_target in categories:
            if category_name not in category_times:
                continue
                
            test_count = category_counts[category_name]
            test_time = category_times[category_name]
            production_target = production_targets[category_name]
            
            if test_count > 0:
                # 1パターンあたりの時間
                time_per_pattern = test_time / test_count
                
                # 本番実行の推定時間
                estimated_time = time_per_pattern * production_target
                total_estimated_time += estimated_time
                
                test_time_str = str(timedelta(seconds=int(test_time)))
                estimated_time_str = str(timedelta(seconds=int(estimated_time)))
                
                print(f"{category_name:12s} | {test_count:8d} | {production_target:8d} | {test_time_str:>12s} | {estimated_time_str:>12s}")
        
        print("-" * 70)
        total_estimated_hours = total_estimated_time / 3600
        total_estimated_timedelta = timedelta(seconds=int(total_estimated_time))
        
        print(f"\n{'合計推定時間':12s} | {'':>8s} | {30000:8d} | {'':>12s} | {str(total_estimated_timedelta):>12s}")
        print(f"\n推定結果:")
        print(f"  - 30,000パターン生成予想時間: {total_estimated_timedelta}")
        print(f"  - 約 {total_estimated_hours:.1f} 時間")
        
        if total_estimated_hours > 10:
            print(f"\n⚠ 警告: 推定時間が長いです。以下を検討してください：")
            print(f"  - より高速なモデルを使用")
            print(f"  - GPU搭載マシンを使用")
            print(f"  - config.pyのBATCH_SIZEを増やす")
        elif total_estimated_hours > 5:
            print(f"\n💡 推定時間は約{total_estimated_hours:.1f}時間です。")
            print(f"   長時間実行となるため、バックグラウンド実行を推奨します。")
    
    print(f"\n出力ファイル: {args.output}")


if __name__ == '__main__':
    main()
