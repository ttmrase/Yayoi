# 合成帳票用フォント(手書き風・SIL OFL 1.1)

フォントファイルはリポジトリに含めない。`python scripts/fetch_fonts.py` で
`$YAYOI_PRIVATE_DATA/fonts/`(環境変数 `YAYOI_FONT_DIR` で変更可)に取得する。
取得元は google/fonts リポジトリ(`ofl/` 配下)。取得時に METADATA.pb のライセンス表記が
OFL であること、OFL.txt の本文が SIL Open Font License 1.1 であることを確認する。

| フォント | 作者 | ライセンス | 取得ファイル | 出所 |
|---|---|---|---|---|
| Yomogi | Satsuyako | SIL Open Font License 1.1 | Yomogi-Regular.ttf | https://github.com/google/fonts/tree/main/ofl/yomogi |
| Zen Kurenaido | Yoshimichi Ohira | SIL Open Font License 1.1 | ZenKurenaido-Regular.ttf | https://github.com/google/fonts/tree/main/ofl/zenkurenaido |
| Klee One | Fontworks Inc. | SIL Open Font License 1.1 | KleeOne-Regular.ttf, KleeOne-SemiBold.ttf | https://github.com/google/fonts/tree/main/ofl/kleeone |
| Hachi Maru Pop | Nonty | SIL Open Font License 1.1 | HachiMaruPop-Regular.ttf | https://github.com/google/fonts/tree/main/ofl/hachimarupop |
| Yusei Magic | Tanukizamurai | SIL Open Font License 1.1 | YuseiMagic-Regular.ttf | https://github.com/google/fonts/tree/main/ofl/yuseimagic |
| Kiwi Maru | Hiroki-Chan | SIL Open Font License 1.1 | KiwiMaru-Light.ttf, KiwiMaru-Regular.ttf, KiwiMaru-Medium.ttf | https://github.com/google/fonts/tree/main/ofl/kiwimaru |
| Zen Maru Gothic | Yoshimichi Ohira | SIL Open Font License 1.1 | ZenMaruGothic-Light.ttf, ZenMaruGothic-Regular.ttf, ZenMaruGothic-Medium.ttf, ZenMaruGothic-Bold.ttf, ZenMaruGothic-Black.ttf | https://github.com/google/fonts/tree/main/ofl/zenmarugothic |

各フォントの著作権表示:

- Yomogi: Copyright 2020 The Yomogi Project Authors (https://github.com/satsuyako/YomogiFont), all rights reserved.
- Zen Kurenaido: Copyright 2021 The Zen Kurenaido Project Authors (https://github.com/googlefonts/zen-kurenaido)
- Klee One: Copyright 2020 The Klee Project Authors (https://github.com/fontworks-fonts/Klee)
- Hachi Maru Pop: Copyright 2020 The Hachi Maru Pop Project Authors (https://github.com/noriokanisawa/HachiMaruPop)
- Yusei Magic: Copyright 2020 The Yusei Magic Project Authors (https://github.com/tanukifont/YuseiMagic)
- Kiwi Maru: Copyright 2020 The Kiwi Maru Project Authors (https://github.com/Kiwi-KawagotoKajiru/Kiwi-Maru)
- Zen Maru Gothic: Copyright 2021 The Zen Maru Gothic Authors (https://github.com/googlefonts/zen-marugothic)

合成帳票は学習・評価用の架空データであり、フォントそのものを再配布しない。

## 取得したファイルの SHA256(2026-10-30 に取得し、合成データの生成に使ったもの)

取得元は google/fonts の `main`(コミットを固定していない)。**再取得したファイルの SHA256 が下と一致しないと、合成データを同じ内容(ビット単位)では再生成できない**
(`scripts/fetch_fonts.py` が `manifest.json` に SHA256 を記録するので、突き合わせる)。フォントが更新されていた場合は、`docs/trial_rc.md` §7 の手順で、重みも作り直す。

| ファイル | バイト数 | SHA256 |
|---|---:|---|
| `HachiMaruPop-Regular.ttf` | 4,385,624 | `78408910c8f1a2f174a279cbc1484b48b71780039eba3fe1be2bfcc5d4df3f98` |
| `KiwiMaru-Light.ttf` | 4,982,684 | `30f856bc944911b025bfbf640bf6b9ffe18a9c7b06b20d4ef26fe5cc9b3819f8` |
| `KiwiMaru-Medium.ttf` | 5,123,612 | `b2659f300a7d48c3f29eb273ffc5e1b26cc416ac8c37ff6bb2f3e43c2f4d235a` |
| `KiwiMaru-Regular.ttf` | 5,065,572 | `b0c3103b2639f690c1fcb44e060058383174bfd2eb72e6635bc9869b374dee87` |
| `KleeOne-Regular.ttf` | 8,724,204 | `bf4063f030cc2ae6adf0a11424a1888e5c0eb4438f1f6d02f52294af868e9b3a` |
| `KleeOne-SemiBold.ttf` | 8,905,128 | `b031ec426c23ca1143ef1f7d58bee7a79efe119ed654152f121c922202b303fd` |
| `Yomogi-Regular.ttf` | 4,045,904 | `ec280f473a03187292905618f24fdb08a6dcbf8620b52637e3b589fe745259b9` |
| `YuseiMagic-Regular.ttf` | 3,134,968 | `82098615f39ed9da6a8ccc674b9006e49c70dd5b775a7a1697f6bedd22ce25a2` |
| `ZenKurenaido-Regular.ttf` | 4,303,112 | `58b8d930d9fc10c8a5810c085bae378dacb98d0779073ee6d53d919f19ee6a4f` |
| `ZenMaruGothic-Black.ttf` | 3,710,324 | `6bd74fe76cd39ee0ec18775c3661d845343fb3f6f8fa09a3076638417baf741f` |
| `ZenMaruGothic-Bold.ttf` | 3,779,008 | `fe24426b9c8b5523a0146a8235c8674eccf0493af354a53ec895c3596d9eb745` |
| `ZenMaruGothic-Light.ttf` | 3,723,392 | `c4e6fa10ff517df37cb0c13038fe2813f3c47afe6577fa4a5573200e4508b2c1` |
| `ZenMaruGothic-Medium.ttf` | 3,810,988 | `3cfdb98a13571ede17fcc769f5093a97c38b80a7b9b2ab754a26b4d822092b3b` |
| `ZenMaruGothic-Regular.ttf` | 3,832,756 | `a0c0b53543e0993ae2225e629c833f3d51495ad31720694ff112ce4ce11111ef` |

学習・較正のデータには、`YuseiMagic`・`KleeOne` を除く 11 書体を使った(`--font-exclude yusei klee`。この 2 書体は評価でだけ使う)。
