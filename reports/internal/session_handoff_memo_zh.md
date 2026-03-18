# SPBCE 工作階段備忘錄

最後更新時間：2026-03-18 Asia/Taipei

## 目前狀態

- 目前主線已收斂為 `deepseek_direct -> heuristic fallback`
- `formal50 v1`、`frozen v2_100`、`frozen v3_250` 都已建立且通過 anti-leakage audit
- `v3_250` 的正式 DeepSeek-only benchmark 已完成，結果支持：
  - `deepseek_direct` 應作為正式 MVP 主模型
  - `heuristic` 應作為 fallback
  - `combiner / hybrid` 不再是當前主線

## 這次新增內容

- 新增中文內部分析報告生成腳本：
  - `scripts/generate_internal_report_zh.py`
- 重新生成中文 HTML 報告：
  - `reports/internal/spbce_internal_analysis_report.html`
  - `reports/internal/spbce_internal_analysis_report_zh.html`
- 新增 handoff 備忘錄：
  - `reports/internal/session_handoff_memo_zh.md`

## 報告內容重點

- 資料來源與用途
- canonical schema / paired behavior contract 的定位
- anti-leakage 與資料汙染控制
- formal50 / v2 / v3 benchmark 完整比較
- behavior benchmark 結果
- DeepSeek / heuristic / combiner 的階段性決策
- frozen v3 slice analysis
- 目前 MVP inference 路徑與產品建議

## 目前最重要結論

1. `DeepSeek direct` 是目前最合理的 MVP 主模型。
2. `heuristic` 保留作 fallback。
3. `combiner / hybrid` 目前沒有穩定商業價值，應停止作為主線。
4. `behavior validity` 目前仍不能對外宣稱 AI 已優於人類 survey 預測真實商業 outcome。

## 若在新視窗接續工作，建議先做的事

1. 先確認目前分支與最新 commit：
   - `git branch --show-current`
   - `git log --oneline -5`
2. 打開以下檔案快速恢復上下文：
   - `reports/internal/spbce_internal_analysis_report_zh.html`
   - `reports/benchmarks/gss_prompt_benchmark_deepseek_chat_v3_250_comparison.json`
   - `reports/benchmarks/gss_prompt_benchmark_deepseek_chat_v3_250_slices.json`
   - `reports/audits/formal_prompt_holdout_v3_250_audit.json`
3. 如果下一步要走產品化，優先順序應是：
   - auth
   - rate limiting
   - persistent logging
   - monitoring
   - request tracing
   - cost dashboard
   - API docs

## 可直接複製到新視窗的接續提示

```text
你現在在 D:\\dev\\Gnosis 專案中，請先承接目前已完成的 SPBCE 工作狀態，不要重新探索整個 repo。

先讀這些檔案恢復上下文：
1. D:\\dev\\Gnosis\\reports\\internal\\spbce_internal_analysis_report_zh.html
2. D:\\dev\\Gnosis\\reports\\benchmarks\\gss_prompt_benchmark_deepseek_chat_v3_250_comparison.json
3. D:\\dev\\Gnosis\\reports\\benchmarks\\gss_prompt_benchmark_deepseek_chat_v3_250_slices.json
4. D:\\dev\\Gnosis\\reports\\audits\\formal_prompt_holdout_v3_250_audit.json
5. D:\\dev\\Gnosis\\src\\spbce\\inference\\mvp.py

目前已知結論：
- MVP 主模型是 deepseek_direct
- fallback 是 heuristic
- frozen v3_250 benchmark 支持 DeepSeek direct 作為主路徑
- combiner / hybrid 不再是主線
- behavior validity 還不能對外宣稱 superiority

請先根據上述狀態，整理「下一個最合理的產品化階段」，不要回頭再做 MiniMax、persona baseline、或 combiner 研究。
```
