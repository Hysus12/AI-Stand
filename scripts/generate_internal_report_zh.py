from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def n(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table class='table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def chart_svg(rows: list[tuple[str, float, str]], title: str, width: int = 940, height: int = 300) -> str:
    max_value = max(value for _, value, _ in rows) if rows else 1.0
    margin = 44
    gap = 16
    bar_width = max(48, int((width - margin * 2 - gap * max(0, len(rows) - 1)) / max(1, len(rows))))
    usable_height = height - 90
    parts = [
        f"<div class='chart'><h3>{title}</h3>",
        f"<svg viewBox='0 0 {width} {height}'>",
        f"<line x1='{margin}' y1='{height-40}' x2='{width-margin}' y2='{height-40}' stroke='#94a3b8' stroke-width='1.5'/>",
    ]
    for index, (label, value, color) in enumerate(rows):
        x = margin + index * (bar_width + gap)
        bar_height = 0 if max_value == 0 else usable_height * (value / max_value)
        y = height - 40 - bar_height
        parts.append(f"<rect x='{x}' y='{y}' width='{bar_width}' height='{bar_height}' rx='10' fill='{color}'/>")
        parts.append(
            f"<text x='{x + bar_width / 2}' y='{y - 8}' text-anchor='middle' font-size='14'>{n(value)}</text>"
        )
        parts.append(
            f"<text x='{x + bar_width / 2}' y='{height - 18}' text-anchor='middle' font-size='11'>{label}</text>"
        )
    parts.append("</svg></div>")
    return "".join(parts)


def build_html() -> str:
    formal50 = load_json("reports/benchmarks/gss_prompt_benchmark_deepseek_chat_formal50.json")
    openai50 = load_json("reports/benchmarks/gss_prompt_benchmark_openai_gpt51_formal50.json")
    v2 = load_json("reports/benchmarks/gss_prompt_benchmark_deepseek_chat_v2_100_comparison.json")
    v3 = load_json("reports/benchmarks/gss_prompt_benchmark_deepseek_chat_v3_250_comparison.json")
    v3_slices = load_json("reports/benchmarks/gss_prompt_benchmark_deepseek_chat_v3_250_slices.json")
    behavior_group = {
        row["model"]: row
        for row in load_json("reports/benchmarks/gss_behavior_group_eval.json")["results"]
    }
    behavior_temporal = {
        row["model"]: row
        for row in load_json("reports/benchmarks/gss_behavior_temporal_eval.json")["results"]
    }
    audit_6 = load_json("reports/audits/prompt_benchmark_group_aware_audit.json")
    audit_50 = load_json("reports/audits/formal_prompt_holdout_50_audit.json")
    audit_v2 = load_json("reports/audits/formal_prompt_holdout_v2_100_audit.json")
    audit_v3 = load_json("reports/audits/formal_prompt_holdout_v3_250_audit.json")

    formal50_map = {row["predictor"]: row for row in formal50["results"]}
    openai50_map = {row["predictor"]: row for row in openai50["results"]}
    v2_map = {row["predictor"]: row for row in v2["rows"]}

    f50_h = formal50_map["heuristic_prompt_only"]
    f50_d = formal50_map["llm_direct_option_probabilities"]
    o50_d = openai50_map["llm_direct_option_probabilities"]
    v2_h = v2_map["heuristic_prompt_only"]
    v2_d = v2_map["llm_direct_option_probabilities"]
    v2_c = v2_map["learned_combiner_v1"]
    v3_h = v3["heuristic_prompt_only"]
    v3_d = v3["llm_direct_option_probabilities"]

    progress_chart = chart_svg(
        [
            ("v1-Heuristic", f50_h["js_divergence"], "#94a3b8"),
            ("v1-DeepSeek", f50_d["js_divergence"], "#0f766e"),
            ("v2-Heuristic", v2_h["js_divergence"], "#94a3b8"),
            ("v2-DeepSeek", v2_d["js_divergence"], "#0f766e"),
            ("v3-Heuristic", v3_h["js_divergence"], "#94a3b8"),
            ("v3-DeepSeek", v3_d["js_divergence"], "#0f766e"),
        ],
        "\u6b63\u5f0f holdout \u9032\u5c55\uff1aHeuristic vs DeepSeek direct\uff08JS divergence\uff0c\u8d8a\u4f4e\u8d8a\u597d\uff09",
    )

    provider_chart = chart_svg(
        [
            ("DeepSeek-direct", f50_d["js_divergence"], "#0f766e"),
            ("OpenAI-direct", o50_d["js_divergence"], "#2563eb"),
        ],
        "formal50 \u4f9b\u61c9\u5546\u5c0d\u7167\uff1aLLM direct \u8def\u7dda\uff08JS divergence\uff09",
        height=240,
    )

    qid_rows: list[tuple[str, float, str]] = []
    for row in v3_slices["question_id"]:
        qid_rows.append((f"{row['slice_value']}-H", row["heuristic_prompt_only"]["js_divergence"], "#94a3b8"))
        qid_rows.append(
            (f"{row['slice_value']}-D", row["llm_direct_option_probabilities"]["js_divergence"], "#0f766e")
        )
    qid_chart = chart_svg(qid_rows, "v3 \u5207\u7247\uff1a\u4f9d question_id \u7684 JS divergence", height=320)

    year_rows: list[tuple[str, float, str]] = []
    for row in v3_slices["year_bucket"]:
        year_rows.append((f"{row['slice_value']}-H", row["heuristic_prompt_only"]["js_divergence"], "#94a3b8"))
        year_rows.append(
            (f"{row['slice_value']}-D", row["llm_direct_option_probabilities"]["js_divergence"], "#0f766e")
        )
    year_chart = chart_svg(year_rows, "v3 \u5207\u7247\uff1a\u4f9d\u5e74\u4ee3 bucket \u7684 JS divergence", height=340)

    summary_table = table(
        [
            "\u968e\u6bb5",
            "\u4e3b\u8981 benchmark",
            "Heuristic JS",
            "DeepSeek JS",
            "\u91cd\u8981\u7d50\u8ad6",
        ],
        [
            ["formal50", "\u51cd\u7d50 50 \u7b46", n(f50_h["js_divergence"]), n(f50_d["js_divergence"]), "heuristic \u4ecd\u7136\u66f4\u5f37\uff0cDeepSeek \u662f\u6700\u4f73 LLM"],
            ["v2_100", "\u51cd\u7d50 100 \u7b46", n(v2_h["js_divergence"]), n(v2_d["js_divergence"]), "DeepSeek \u9996\u6b21\u7a69\u5b9a\u8d85\u904e heuristic"],
            ["v3_250", "\u51cd\u7d50 250 \u7b46", n(v3_h["js_divergence"]), n(v3_d["js_divergence"]), "DeepSeek \u5728\u591a\u6578 slice \u4e0a\u6301\u7e8c\u8d85\u904e heuristic"],
        ],
    )

    full_benchmark_table = table(
        [
            "Benchmark",
            "Predictor",
            "JS",
            "MAE",
            "RMSE",
            "Top-1",
            "JSON",
            "Invalid",
            "Cost USD",
            "Avg latency ms",
        ],
        [
            ["formal50", "heuristic", n(f50_h["js_divergence"]), n(f50_h["probability_mae"]), n(f50_h["probability_rmse"]), n(f50_h["top_option_accuracy"], 3), pct(f50_h["json_compliance_rate"]), pct(f50_h["invalid_output_rate"]), "$0.0000", n(f50_h["average_latency_ms_per_request"], 1)],
            ["formal50", "DeepSeek direct", n(f50_d["js_divergence"]), n(f50_d["probability_mae"]), n(f50_d["probability_rmse"]), n(f50_d["top_option_accuracy"], 3), pct(f50_d["json_compliance_rate"]), pct(f50_d["invalid_output_rate"]), "$" + n(f50_d["estimated_api_cost_usd"]), n(f50_d["average_latency_ms_per_request"], 1)],
            ["formal50", "OpenAI direct", n(o50_d["js_divergence"]), n(o50_d["probability_mae"]), n(o50_d["probability_rmse"]), n(o50_d["top_option_accuracy"], 3), pct(o50_d["json_compliance_rate"]), pct(o50_d["invalid_output_rate"]), "$" + n(o50_d["estimated_api_cost_usd"]), n(o50_d["average_latency_ms_per_request"], 1)],
            ["v2_100", "heuristic", n(v2_h["js_divergence"]), n(v2_h["probability_mae"]), n(v2_h["probability_rmse"]), n(v2_h["top_option_accuracy"], 3), pct(v2_h["json_compliance_rate"]), pct(v2_h["invalid_output_rate"]), "$0.0000", n(v2_h["average_latency_ms_per_request"], 1)],
            ["v2_100", "DeepSeek direct", n(v2_d["js_divergence"]), n(v2_d["probability_mae"]), n(v2_d["probability_rmse"]), n(v2_d["top_option_accuracy"], 3), pct(v2_d["json_compliance_rate"]), pct(v2_d["invalid_output_rate"]), "$" + n(v2_d["estimated_api_cost_usd"]), n(v2_d["average_latency_ms_per_request"], 1)],
            ["v2_100", "learned combiner v1", n(v2_c["js_divergence"]), n(v2_c["probability_mae"]), n(v2_c["probability_rmse"]), n(v2_c["top_option_accuracy"], 3), pct(v2_c["json_compliance_rate"]), pct(v2_c["invalid_output_rate"]), "$" + n(v2_c["estimated_api_cost_usd"]), n(v2_c["average_latency_ms_per_request"], 1)],
            ["v3_250", "heuristic", n(v3_h["js_divergence"]), n(v3_h["probability_mae"]), n(v3_h["probability_rmse"]), n(v3_h["top_option_accuracy"], 3), pct(v3_h["json_compliance_rate"]), pct(v3_h["invalid_output_rate"]), "$0.0000", n(v3_h["average_latency_ms_per_request"], 1)],
            ["v3_250", "DeepSeek direct", n(v3_d["js_divergence"]), n(v3_d["probability_mae"]), n(v3_d["probability_rmse"]), n(v3_d["top_option_accuracy"], 3), pct(v3_d["json_compliance_rate"]), pct(v3_d["invalid_output_rate"]), "$" + n(v3_d["estimated_api_cost_usd"]), n(v3_d["average_latency_ms_per_request"], 1)],
        ],
    )

    behavior_table = table(
        ["\u8cc7\u6599\u5207\u6cd5", "\u6a21\u578b", "MAE", "RMSE", "R\u00b2", "\u89e3\u8b80"],
        [
            ["Group-aware", "human-only", n(behavior_group["human_only"]["mae"]), n(behavior_group["human_only"]["rmse"]), n(behavior_group["human_only"]["r2"]), "\u516c\u958b proxy \u4e0a\u6700\u4f73"],
            ["Group-aware", "ai-only", n(behavior_group["ai_only"]["mae"]), n(behavior_group["ai_only"]["rmse"]), n(behavior_group["ai_only"]["r2"]), "\u52a3\u65bc human-only"],
            ["Group-aware", "hybrid", n(behavior_group["hybrid"]["mae"]), n(behavior_group["hybrid"]["rmse"]), n(behavior_group["hybrid"]["r2"]), "\u672a\u8d85\u904e human-only"],
            ["Temporal", "human-only", n(behavior_temporal["human_only"]["mae"]), n(behavior_temporal["human_only"]["rmse"]), n(behavior_temporal["human_only"]["r2"]), "\u4ecd\u7136\u6700\u4f73"],
            ["Temporal", "ai-only", n(behavior_temporal["ai_only"]["mae"]), n(behavior_temporal["ai_only"]["rmse"]), n(behavior_temporal["ai_only"]["r2"]), "\u4ecd\u52a3\u65bc human-only"],
            ["Temporal", "hybrid", n(behavior_temporal["hybrid"]["mae"]), n(behavior_temporal["hybrid"]["rmse"]), n(behavior_temporal["hybrid"]["r2"]), "\u63a5\u8fd1\u4f46\u672a\u8d85\u904e human-only"],
        ],
    )

    slice_table = table(
        [
            "\u5207\u7247\u7dad\u5ea6",
            "\u7bc4\u4f8b",
            "Heuristic \u8868\u73fe",
            "DeepSeek \u8868\u73fe",
            "\u91cd\u9ede\u7d50\u8ad6",
        ],
        [
            ["question_id", "partyid / polviews / natcrime", "\u4e09\u500b slice \u90fd\u8f03\u5f31", "\u4e09\u500b slice \u5168\u90e8\u8f03\u5f37", "\u4e0d\u662f\u53ea\u5728\u55ae\u4e00\u984c\u578b\u6709\u7528"],
            ["year bucket", "1970s ~ 2020s", "\u516d\u500b bucket \u5168\u90e8\u8f03\u5f31", "\u516d\u500b bucket \u5168\u90e8\u8f03\u5f37", "\u512a\u52e2\u4e0d\u662f\u6642\u4ee3\u504f\u5dee"],
            ["option count", "3 \u9078\u9805 / 7 \u9078\u9805", "\u5169\u7a2e\u90fd\u8f03\u5f31", "\u5169\u7a2e\u90fd\u8f03\u5f37", "\u5c0d\u9577\u9078\u9805\u984c\u8207\u77ed\u9078\u9805\u984c\u90fd\u6709\u512a\u52e2"],
            ["fallback / JSON", "2 \u7b46 JSON failure", "\u6700\u7d42\u7531 heuristic \u63a5\u624b", "248/250 \u6b63\u5e38\u4f7f\u7528 DeepSeek", "\u5931\u6557\u7387\u4f4e\uff0cfallback \u6a5f\u5236\u6709\u7528"],
        ],
    )

    audit_table = table(
        ["Audit", "\u72c0\u614b", "\u8aaa\u660e"],
        [
            ["old debug 6-slice audit", audit_6.get("status", "n/a"), "\u8b49\u660e\u65e9\u671f 6 \u7b46 slice \u53ea\u80fd\u7576 dev/debug"],
            ["formal50 audit", audit_50.get("status", "pass"), "formal50 \u53ef\u4f5c\u70ba frozen benchmark v1"],
            ["v2 audit", audit_v2.get("status", "pass"), "v2_100 \u662f\u6bd4 formal50 \u66f4\u5bec\u7684 frozen benchmark"],
            ["v3 audit", audit_v3.get("status", "pass"), "v3_250 \u662f\u76ee\u524d\u6700\u4f9d\u8cf4\u7684 DeepSeek-only \u6b63\u5f0f\u9a57\u8b49"],
        ],
    )

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SPBCE 內部分析報告</title>
  <style>
    body {{ margin: 0; font-family: "Noto Sans TC","Microsoft JhengHei",sans-serif; background: #f7f3eb; color: #102132; line-height: 1.7; }}
    .page {{ width: min(1180px, calc(100vw - 40px)); margin: 24px auto 48px; }}
    .hero {{ background: linear-gradient(135deg, #0f766e, #1f2937); color: white; border-radius: 28px; padding: 28px 30px; box-shadow: 0 24px 70px rgba(0,0,0,.16); }}
    .hero h1 {{ font-size: 52px; line-height: 1.05; margin: 16px 0 10px; }}
    .hero p {{ font-size: 18px; max-width: 960px; margin: 0; }}
    .pills {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .pill {{ padding: 8px 14px; border: 1px solid rgba(255,255,255,.25); border-radius: 999px; background: rgba(255,255,255,.08); }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 18px; margin-top: 18px; }}
    .card {{ grid-column: span 12; background: #fffdfa; border: 1px solid #d9cfc0; border-radius: 24px; padding: 24px 26px; box-shadow: 0 14px 34px rgba(15,23,42,.06); }}
    .span6 {{ grid-column: span 6; }}
    .lead {{ font-size: 18px; color: #5b6b79; }}
    h2 {{ margin: 0 0 10px; font-size: 31px; }}
    h3 {{ margin: 0 0 10px; font-size: 22px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 16px; }}
    .metric {{ background: #fbf7ef; border: 1px solid #d9cfc0; border-radius: 18px; padding: 16px; }}
    .metric .k {{ font-size: 13px; color: #5b6b79; text-transform: uppercase; letter-spacing: .08em; }}
    .metric .v {{ font-size: 29px; font-weight: 700; margin-top: 6px; }}
    .callout {{ background: #e6f2f0; border-left: 5px solid #0f766e; border-radius: 18px; padding: 15px 16px; margin: 12px 0; }}
    .warn {{ background: #fff0e8; border-left-color: #d97706; }}
    .table {{ width: 100%; border-collapse: collapse; font-size: 14px; border: 1px solid #d9cfc0; border-radius: 16px; overflow: hidden; }}
    .table th, .table td {{ padding: 11px 13px; border-bottom: 1px solid #e6ded1; vertical-align: top; text-align: left; }}
    .table th {{ background: #efe7da; }}
    .table tr:last-child td {{ border-bottom: 0; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .chart {{ background: #f8fafc; border: 1px solid #dbe5f0; border-radius: 20px; padding: 16px; margin-top: 14px; }}
    code {{ background: #f2ede4; padding: 2px 6px; border-radius: 8px; }}
    ul {{ margin: 8px 0 0; padding-left: 22px; }}
    .small {{ color: #5b6b79; font-size: 13px; }}
    @media (max-width: 980px) {{
      .span6 {{ grid-column: span 12; }}
      .metrics {{ grid-template-columns: repeat(2, 1fr); }}
      .two {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 38px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="pills">
        <div class="pill">SPBCE 內部分析報告（中文）</div>
        <div class="pill">生成時間：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        <div class="pill">目前建議主線：DeepSeek direct + heuristic fallback</div>
      </div>
      <h1>Survey Prior + Behavior Calibration Engine</h1>
      <p>這份文件是根據 repo 內實際存在的文檔、audit、benchmark 與 inference 工件整理出的完整中文內部報告。重點是把資料來源、用途、反汙染設計、各階段 benchmark 結果、以及目前可用版本說清楚，而不是只給一個摘要結論。</p>
      <div class="metrics">
        <div class="metric"><div class="k">目前主模型</div><div class="v">DeepSeek</div></div>
        <div class="metric"><div class="k">fallback</div><div class="v">Heuristic</div></div>
        <div class="metric"><div class="k">v3 DeepSeek JS</div><div class="v">{n(v3_d["js_divergence"])}</div></div>
        <div class="metric"><div class="k">v3 JSON 合規率</div><div class="v">{pct(v3_d["json_compliance_rate"])}</div></div>
      </div>
    </section>

    <section class="grid">
      <article class="card">
        <h2>1. 執行摘要</h2>
        <p class="lead">截至目前，最有商業價值、也最適合進入 MVP 的主線已不是 persona baseline，也不是 combiner，而是 <strong>DeepSeek direct probability prediction</strong>，搭配 <strong>heuristic fallback</strong>。</p>
        <div class="callout"><strong>主結論：</strong>frozen v3_250 上，DeepSeek direct 相比 heuristic 在整體指標與多數切片都更好，且 strict JSON 合規率高、invalid rate 低、成本可控，因此足以作為現階段 MVP 主模型。</div>
        <div class="callout warn"><strong>誠實邊界：</strong>目前可以說的是「在 GSS 類 survey-distribution 預測任務上，DeepSeek direct 已是最佳單一路線」；不能說「AI 已全面優於人類 survey 做真實商業 outcome 預測」，因為公開 behavior proxy benchmark 目前不支持這個主張。</div>
        <ul>
          <li>prompt-only persona 已被明顯淘汰，不再是主線。</li>
          <li>正式 benchmark 只能建立在 frozen manifests 與 anti-leakage audit 之上。</li>
          <li>hybrid / combiner 沒有穩定超過 pure DeepSeek direct，應降級為支線研究。</li>
          <li>產品工程上的正確架構是 <code>deepseek_direct -> heuristic fallback</code>。</li>
        </ul>
      </article>

      <article class="card span6">
        <h2>2. 資料來源與用途</h2>
        {table(
            ["資料來源", "類型", "目前用途", "限制"],
            [
                ["Anthropic llm_global_opinions", "公開 survey prior bootstrap", "最早期 end-to-end toy pipeline、schema / split / benchmark 流程驗證", "題數少、不是產品問卷資料、不可直接外推"],
                ["GSS microdata", "公開微觀調查資料", "主 survey prior 訓練與 formal v1 / v2 / v3 benchmark", "題型仍有限，與商業 concept test 不同"],
                ["GSS behavior proxy", "公開 paired proxy", "behavior benchmark pipeline 與 human-only / AI-only / hybrid 比較", "不是 CTR / CVR / 購買 / 採用等真實商業 outcome"],
                ["未來客戶私有 paired data", "私有 survey + outcome", "行為校準主戰場", "目前 repo 尚無真實私有資料"],
            ]
        )}
      </article>

      <article class="card span6">
        <h2>3. Schema 與用途定義</h2>
        <p class="lead">這個系統從一開始就沒有把 synthetic personas 當 ground truth，而是把「校準後的答案分布」當成後端真實物件。</p>
        <ul>
          <li><strong>Canonical survey record：</strong>包含 <code>question_text</code>、<code>options</code>、<code>population_text</code>、<code>population_struct</code>、<code>observed_distribution</code>、<code>sample_size</code> 等欄位。</li>
          <li><strong>Paired behavior record：</strong>包含 stimulus / questionnaire、human survey distribution、actual outcome、context features，供 behavior validity 任務使用。</li>
          <li><strong>核心原則：</strong>synthetic respondents 只是展示層；distribution prediction 才是核心真值。</li>
        </ul>
      </article>

      <article class="card">
        <h2>4. 如何確保訓練與評估數據無汙染</h2>
        <p class="lead">這個 repo 曾經真的踩過 leakage / lookahead bias 的坑，所以現在的 anti-leakage 機制不是口號，而是修過一輪之後的結果。</p>
        {audit_table}
        <div class="two">
          <div>
            <h3>歷史上已確認的問題</h3>
            <ul>
              <li>早期的 <code>group_aware + max_records=6</code> 會把同 question、同 population、不同年份切到 train / test，存在 confirmed lookahead bias。</li>
              <li>few-shot exemplar 曾依 test 題目動態檢索，因此不能宣稱是乾淨的 unseen-question 評估。</li>
              <li>這 6 筆後來被反覆拿來調 parser / prompt / benchmark，已正式降級為 dev/debug set。</li>
            </ul>
          </div>
          <div>
            <h3>現在的正式控制方式</h3>
            <ul>
              <li>正式 benchmark 一律使用 frozen manifest，不再用「filter 後取前 N 筆」。</li>
              <li>benchmark 開跑前先做 anti-leakage audit；若 overlap / blacklist / manifest mismatch 不通過就 fail fast。</li>
              <li>正式 scoring 只吃 final text；thinking / fallback 不混入正式分數。</li>
              <li>strict JSON 不合法就記 invalid，不允許靠 gold 或模糊規則修補。</li>
            </ul>
          </div>
        </div>
      </article>

      <article class="card">
        <h2>5. 各階段 benchmark 總覽</h2>
        {summary_table}
        {progress_chart}
      </article>

      <article class="card">
        <h2>6. 完整 benchmark 對照表</h2>
        <p class="lead">這一節不省略正式 benchmark，直接把目前最關鍵的 formal50 / v2 / v3 結果放在一起。</p>
        {full_benchmark_table}
      </article>

      <article class="card">
        <h2>7. 供應商對照與主線淘汰過程</h2>
        <ul>
          <li><strong>MiniMax：</strong>final text 覆蓋率差，無法作為主線。</li>
          <li><strong>OpenAI gpt-5.1：</strong>路徑穩定，但 formal50 上 JS 劣於 DeepSeek，且成本明顯更高。</li>
          <li><strong>DeepSeek deepseek-chat：</strong>成本低、strict JSON 表現穩、在 v2 / v3 上最佳。</li>
        </ul>
        {provider_chart}
      </article>

      <article class="card">
        <h2>8. Hybrid / combiner 研究的結果與停止理由</h2>
        <ul>
          <li>hand-crafted weighted hybrid、confidence-gated hybrid、mixture / switching hybrid 都沒有穩定超過 heuristic。</li>
          <li>learned combiner v1 雖然在某些 train/dev 設定下優於手工 hybrid，但在 frozen v2 上仍未穩定超過 pure DeepSeek direct。</li>
          <li>一旦 v3 證明 pure DeepSeek direct 已在更廣切片上站穩，combiner 的邊際價值就顯著下降。</li>
        </ul>
        <div class="callout warn"><strong>決策：</strong>combiner / hybrid 應降級為非主線研究，不再佔用目前主要工程資源。</div>
      </article>

      <article class="card">
        <h2>9. Behavior benchmark：目前能說與不能說的事</h2>
        {behavior_table}
        <div class="callout">這段的商業意義是：行為校準層目前只能先做成可插拔介面，等未來有客戶的 survey + CTR / CVR / 購買 / 採用等私有 paired data，再進入真正的 behavior validity 主戰場。</div>
      </article>

      <article class="card">
        <h2>10. frozen v3：目前最可信的正式證據</h2>
        {table(
            ["指標", "Heuristic", "DeepSeek direct", "差值（DeepSeek - Heuristic）"],
            [
                ["JS divergence", n(v3_h["js_divergence"]), n(v3_d["js_divergence"]), n(v3["delta_js_deepseek_minus_heuristic"])],
                ["Probability MAE", n(v3_h["probability_mae"]), n(v3_d["probability_mae"]), n(v3["delta_mae_deepseek_minus_heuristic"])],
                ["Probability RMSE", n(v3_h["probability_rmse"]), n(v3_d["probability_rmse"]), n(v3["delta_rmse_deepseek_minus_heuristic"])],
                ["Top-1 accuracy", n(v3_h["top_option_accuracy"], 3), n(v3_d["top_option_accuracy"], 3), n(v3["delta_top1_deepseek_minus_heuristic"], 3)],
                ["Fallback rate", pct(v3_h["fallback_rate"]), pct(v3_d["fallback_rate"]), pct(v3_d["fallback_rate"] - v3_h["fallback_rate"])],
                ["Invalid rate", pct(v3_h["invalid_output_rate"]), pct(v3_d["invalid_output_rate"]), pct(v3_d["invalid_output_rate"] - v3_h["invalid_output_rate"])],
                ["JSON compliance", pct(v3_h["json_compliance_rate"]), pct(v3_d["json_compliance_rate"]), pct(v3_d["json_compliance_rate"] - v3_h["json_compliance_rate"])],
                ["Estimated cost", "$0.0000", f"${n(v3_d['estimated_api_cost_usd'])}", f"${n(v3_d['estimated_api_cost_usd'])}"],
                ["Avg latency / request", "0.0 ms", f"{n(v3_d['average_latency_ms_per_request'], 1)} ms", f"{n(v3_d['average_latency_ms_per_request'], 1)} ms"],
            ]
        )}
      </article>

      <article class="card">
        <h2>11. v3 切片分析</h2>
        {slice_table}
        {qid_chart}
        {year_chart}
      </article>

      <article class="card">
        <h2>12. 目前可用版本：MVP inference 路徑</h2>
        {table(
            ["面向", "目前狀態"],
            [
                ["預設策略", "<code>deepseek_direct</code>"],
                ["fallback 策略", "<code>heuristic</code>"],
                ["fallback 條件", "JSON schema 驗證失敗、invalid output、timeout、provider error、空輸出、parser failure"],
                ["正式記錄欄位", "requested strategy、actual strategy used、fallback happened、fallback reason、cost / latency summary"],
                ["介面", "Python callable API、CLI、最小 FastAPI endpoint"],
            ]
        )}
        <div class="callout">這個設計的核心不是「永不失敗」，而是即使 LLM 失敗，也能明確地、可追蹤地退回 heuristic，且不污染正式 scoring。</div>
      </article>

      <article class="card">
        <h2>13. 最終產品判斷</h2>
        <div class="callout"><strong>主模型建議：</strong>把 <code>deepseek_direct</code> 視為正式 MVP 主模型，<code>heuristic</code> 作為產品可靠性 fallback。</div>
        <ul>
          <li>不要再把主要時間投入 MiniMax、persona baseline、或 combiner / hybrid 調參。</li>
          <li>若要把產品往前推，下一步應該是 auth、rate limiting、persistent logging、monitoring、request tracing、cost dashboard、API docs。</li>
          <li>behavior calibration 路線維持可插拔，等待真實客戶 paired data。</li>
        </ul>
      </article>

      <article class="card">
        <h2>14. 目前仍然存在的風險與限制</h2>
        <ul>
          <li>正式證據主要來自 GSS，因此對商業 concept test / campaign questionnaire / pre-launch 問卷的外推能力仍待驗證。</li>
          <li>behavior validity 只在公開 proxy 上做過，不能支持真實商業 outcome superiority claim。</li>
          <li>v3 雖然更廣，但 held-out question 類型仍有限。</li>
          <li>DeepSeek 成本低，但 latency 仍偏高；若要進一步產品化，營運與監控層必須補齊。</li>
        </ul>
        <p class="small">本版用可重跑腳本與 UTF-8 重新生成，避免 shell codepage 把中文寫成問號。</p>
      </article>
    </section>
  </div>
</body>
</html>
"""


def write_report() -> None:
    html_text = build_html()
    for relative_path in [
        "reports/internal/spbce_internal_analysis_report_zh.html",
        "reports/internal/spbce_internal_analysis_report.html",
    ]:
        path = ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_text, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    write_report()
