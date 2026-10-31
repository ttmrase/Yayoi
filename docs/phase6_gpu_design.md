# GPU(CUDA 版の PyTorch)に関する設計メモ

**結論: 試用は CPU 版で行う。GPU 版の配布物は、試用の結果(学習の所要時間・利用者の PC の GPU の有無)を見てから判断する。** ここは判断材料の整理で、実装はしない。

## いまの状態

- 配布物の PyTorch は **CPU 版**(PyPI の Windows 向けの wheel。`requirements/trial-win-cp311.txt` で固定。`make_trial_lock.py` が CPU 版であることを確認する)。
- 認識(推論)は CPU で 1 ページ 10〜30 秒(2〜4 スレッド)。追加学習は `app/train_worker.py` が、CPU のスレッドを 2〜4 に絞り、低優先度で動かす。
  GPU があれば使う経路はコードにあるが(`parseq/finetune.py` の `device`、環境変数 `YAYOI_TRAIN_DEVICE`)、**開発機は CPU のみで未検証**(`docs/windows_checklist.md` T8)。
- 学習の所要時間(CPU、2〜4 スレッド)は、確定ページ数に対して数分〜十数分(合成の書き手での計測。`docs/phase4_design.md`)。バックグラウンドで動くので、確認の作業は止まらない。

## GPU 版を配布物に入れるときの論点

| 論点 | 内容 |
|---|---|
| 大きさ | CUDA 版の torch は wheel だけで約 2.5〜3 GB(cuDNN・cuBLAS などを含む)。いまの配布物(約 600 MB 前後)の数倍になる。CPU 版と**別の配布物**にするのが妥当(利用者が選ぶ。同じ zip に両方は入れない) |
| ライセンス | CUDA ランタイムの再配布は NVIDIA の EULA に従う(wheel に含まれる形での再配布は許されているが、帰属・禁止事項を「ライセンス」画面と配布物に追記する)。`scripts/gen_licenses.py` が wheel の METADATA から拾うので、一覧には出る |
| ドライバ | 利用者の NVIDIA ドライバが、wheel の CUDA の版に必要な最低の版を満たす必要がある。満たさないと `torch.cuda.is_available()` が False になる(落ちはしない) |
| 実機での検証 | 開発機に GPU がなく、**GPU の経路は一度も動かしていない**。GPU 版を配るなら、実機(RTX 系を 1〜2 種類)で、推論の数値(CPU との一致の許容範囲)・学習の所要時間・メモリ不足(OOM)のときの CPU への退避を確かめる必要がある |
| 数値の差 | GPU と CPU で推論結果がビット単位では一致しない。確信度のしきい値・較正は CPU(再現可能)で決めてあるので、GPU で推論するときは、確信度が少しずれうる。**採否のゲート・監視は同じデバイスで評価・認識したものを比べる**設計にする(学習のワーカーが評価に使うデバイス = 認識に使うデバイス) |
| 学習だけ GPU | 最初の実装は「学習だけ GPU、推論は CPU のまま」が安全(推論の確信度が変わらない)。学習した重みは CPU でも読める(`torch.load(weights_only=True`、`map_location="cpu"`)。ただし採否の評価は CPU で再認識するので、評価は従来どおりの時間がかかる |
| フォールバック | GPU が使えない・メモリ不足・ドライバの不整合は、**必ず CPU に戻して続ける**(学習の失敗理由コード `device_unavailable` を追加し、学習画面に「CPU で学習しました」と出す)。`YAYOI_TRAIN_DEVICE=cpu` で固定もできる |
| セットアップ | 配布物を分け、`setup.ps1` が選んだ版を入れる(CPU 版の lock と CUDA 版の lock を別に持つ。ハッシュ固定は同じ方式) |

## 試用で確かめて、判断する基準

1. 利用者の PC に NVIDIA の GPU があるか(試用の聞き取り)。
2. CPU での学習の所要時間が「許容できない」か(目安: 確定ページ 50〜100 で 30 分を超える、または確認の作業が重くなる)。
3. 上の 2 つが「GPU あり」かつ「許容できない」なら、GPU 版(学習だけ GPU)の配布物を作る。それ以外は CPU 版のままにする。

## 実装するときの作業(判断が出てから)

1. `requirements/trial-win-cu12x-cp311.in/.txt`(CUDA 版の lock。`make_trial_lock.py` を拡張)と、配布物の種類の切り替え(`make_trial_bundle.py --cuda`)。
2. `train_worker` のデバイス選択の整理(自動 = cuda が使えて空きメモリが十分なら cuda、さもなくば cpu)と、`device_unavailable` のフォールバック・画面の表示。
3. 実機チェックリストに GPU の項目を足す(T8 を具体化)。CI では GPU を使えない(GitHub のホステッドランナーに GPU は無い)ので、CPU との数値一致は自己ホストのランナーか手元の実機で確かめる。
