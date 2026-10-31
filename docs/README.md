# 文書の索引

リポジトリは public。ここにある文書は設計・仕様・手順・記録で、**実データ(帳票の内容)・画像・モデルの重み・正解データは含まない**(置き場所は `YAYOI_PRIVATE_DATA`。ルートの `README.md` 参照)。

## 利用者向け

| 文書 | 内容 |
|---|---|
| [`user_manual.md`](user_manual.md) | 試用版の手引き(初期設定・取り込み・確認のキー操作・CSV の出力と弥生への取り込み・学習画面・バックアップ・困ったとき・アンインストール)。配布物に `manual.html` として入る |
| [`windows_setup.md`](windows_setup.md) | Windows での試用版のセットアップ(オフライン・SHA256 の検証つき) |
| [`yayoi_export.md`](yayoi_export.md) | 弥生インポート形式の仕様と、税区分・出力前の検証 |

## 設計と仕様(段階順)

| 文書 | 内容 |
|---|---|
| [`layout.md`](layout.md) | Phase 1: 表の位置合わせ・セル切り出し |
| [`recognition.md`](recognition.md) | Phase 2: セル種別 → 認識(PARSeq)→ 後処理 → 確信度。処理の流れと評価指標 |
| [`phase3_design.md`](phase3_design.md) | Phase 3: 確認 UI とローカルサーバ(画面・キー操作・セキュリティ・データ領域) |
| [`phase4_design.md`](phase4_design.md) | Phase 4: 世代の管理・追加学習・採用ゲート・監視と自動ロールバック・提案モード |
| [`backup_restore.md`](backup_restore.md) | バックアップと復元(zip の構成・検査・入れ替えの手順・悪意のある zip への備え) |
| [`phase6_packaging.md`](phase6_packaging.md) | Phase 6: 配布・ログ・ライセンス表示・アンインストール・Windows の CI・束の検査 |
| [`phase6_gpu_design.md`](phase6_gpu_design.md) | GPU(CUDA 版)に関する設計メモ(試用版は CPU 版) |

## 検証と計測の記録

| 文書 | 内容 |
|---|---|
| [`writer_adaptation.md`](writer_adaptation.md) | 書き手への追加学習の実現性(Phase 4 の先行検証) |
| [`phase4_method_validation.md`](phase4_method_validation.md) | Phase 4 の手法の検証(天井・R2・R1/R2 の一致による確信度) |
| [`phase3_measurements.md`](phase3_measurements.md) | 確認 UI の計測(操作の遅延・ページの開き・確定)の基準値 |
| [`windows_checklist.md`](windows_checklist.md) | **Windows 実機で確かめる項目**(F' 確認 UI / H バックアップ / I ライセンス・ログ・アンインストール / GPU) |

## 配布物の出所と記録

| 文書 | 内容 |
|---|---|
| [`model_provenance.md`](model_provenance.md) | 同梱する重み 9 ファイルの由来・学習/較正データ・実行したスクリプト・SHA256、実サンプルを使った処理の開示、再検査の手順 |
| [`trial_rc.md`](trial_rc.md) | 試用版の RC の記録(作ったコミット・zip の SHA256・全ファイルの一覧・重みの突き合わせ・検査の結果)と、**重みの作り直しの手順・所要時間・再現性の確認(§7)**。**未公開** |
| [`dependencies.md`](dependencies.md) | 依存ライブラリ・モデルの出所とライセンス |
| [`fonts.md`](fonts.md) | 合成帳票用のフォント(SIL OFL 1.1)の一覧 |
| [`LICENSE-parseq-Apache-2.0.txt`](LICENSE-parseq-Apache-2.0.txt) | PARSeq のライセンス本文 |
| [`ground_truth_schema.json`](ground_truth_schema.json) | 合成帳票の正解 JSON のスキーマ |

## どこから読むか

- **全体をつかむ**: ルートの `README.md`(全体像の図・開発状況)→ `phase3_design.md` → `phase4_design.md`。
- **安全の考え方**(実データを扱わない・外に出さない): ルートの `README.md` の「実データの置き場所」、`phase3_design.md` のセキュリティ、`backup_restore.md` の検査、`model_provenance.md` の開示。
- **Windows の実機で試す人**: `windows_setup.md` → `windows_checklist.md`(結果を記録する項目に T12 の「金額の選択の負担」を含む)。
- **試用版を配る前に**: `trial_rc.md`(束の同一性)→ `model_provenance.md` → `windows_setup.md`(SHA256 を別経路で伝える)。
