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
