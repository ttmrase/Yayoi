import type { TrainingStatus } from "./types";

/** 追加学習の理由コード → 画面の文言。コードはサーバ(app/gate.py・trainers.py・training.py)が返し、値(摘要・金額)は含まない。 */
const REASONS: Record<string, string> = {
  adopted_accuracy_up: "評価用ページで正解率が上がったため採用しました",
  adopted_review_down: "評価用ページで要確認が減ったため採用しました",
  miss_rate_worse: "見逃し(誤りなのに要確認がつかないセル)が増えるため、採用しませんでした",
  accuracy_dropped: "評価用ページで正解率が下がるため、採用しませんでした",
  no_gain: "いまの世代より良くならなかったため、採用しませんでした",
  review_rate_increased: "要確認が増えすぎるため、採用しませんでした",
  eval_too_small: "評価に使えるセルが足りないため、採否を決めずに見送りました(確定ページが増えたら次の機会に)",
  active_generation_changed: "学習中に使用中の世代が変わったため、結果を使わずに見送りました",
  cancelled: "中止しました",
  app_closed: "アプリが閉じられたため中断しました",
  timeout: "時間がかかりすぎたため止めました",
  worker_crashed: "学習のプログラムが異常終了しました",
  no_result: "学習の結果が得られませんでした",
  rehearsal_missing: "忘れ防止用の合成データが見つからないため学習できません",
  pagedata_missing: "学習に使う認識の入力が見つかりません",
  mock_failure: "(開発用)学習の失敗を模擬しました",
};

export const reasonText = (code: string | null): string => {
  if (!code) return "";
  if (REASONS[code]) return REASONS[code];
  return `学習に失敗しました(${code})`;
};

export const STATUS_TEXT: Record<TrainingStatus, string> = {
  running: "学習中",
  adopted: "採用",
  rejected: "不採用",
  deferred: "見送り",
  failed: "失敗",
  interrupted: "中断",
};

/** 開始できない理由のコード(サーバの 4xx)→ 文言 */
export const START_ERROR: Record<string, string> = {
  training_running: "すでに学習中です",
  not_enough_pages: "確定したページが足りません",
  training_disabled: "このバージョンでは追加学習を使えません",
};
