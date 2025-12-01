#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行：从 Excel 数据 + 提示词，调用大模型生成绘图指令并产出图表文件

示例：
  python cli_plot.py --prompt "画出各年份销售额与支出趋势折线图" \
                     --file /path/to/data.xlsx \
                     --output /path/to/result.html \
                     --model deepseek-v3-aliyun \
                     --api-key sk-xxx
"""
import argparse
import os
import sys
import re
import json
from typing import Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gpt_data import llm_model2, llm_text2
from xl_class import write_df_to_excel
from auto_code import check_code_safety, run_generated_code


class OutputManager:
    def __init__(self, output_path: str):
        self.base_path = output_path
        self.counter = 0

        base, ext = os.path.splitext(output_path)
        self.base = base
        self.ext = ext.lower()

        if not self.ext:
            # 默认输出 html
            self.ext = ".html"

    def next_path(self) -> str:
        if self.counter == 0:
            path = f"{self.base}{self.ext}"
        else:
            path = f"{self.base}_{self.counter}{self.ext}"
        self.counter += 1
        return path


def build_df_preview_markdown(
    df: pd.DataFrame,
    max_rows: int = 60,
    max_cols: int = 12,
    select_columns: Optional[str] = None,
    max_cell_chars: int = 120,
):
    """
    将 DataFrame 压缩成“列类型 + 前N行预览”的小体量 Markdown，避免超长上下文。
    - select_columns: 逗号分隔的列名白名单（优先采用）；为空则取前 max_cols 列
    - max_cell_chars: 每个单元格截断长度，降低 token 占用
    """
    cols = None
    if select_columns:
        whitelist = [c.strip() for c in select_columns.split(",") if c.strip()]
        cols = [c for c in whitelist if c in df.columns]
        if not cols:
            cols = list(df.columns[: max_cols])
    else:
        cols = list(df.columns[: max_cols])

    # 列类型表
    dtypes_df = pd.DataFrame({"column": cols, "dtype": [str(df[c].dtype) for c in cols]})
    dtypes_md = dtypes_df.to_markdown(index=False)

    # 预览数据（前 N 行）
    preview = df.loc[:, cols].head(max_rows).copy()
    # 数值列适当四舍五入，降低字符长度
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        try:
            preview[c] = pd.to_numeric(preview[c], errors="coerce").round(6)
        except Exception:
            pass
    # 截断每个单元格，避免超长字符串
    preview = preview.astype(str).applymap(lambda x: (x[:max_cell_chars] + "…") if len(x) > max_cell_chars else x)
    preview_md = preview.to_markdown(index=False)

    meta = f"总行数: {len(df)}，总列数: {df.shape[1]}；预览行数: {min(max_rows, len(df))}，预览列数: {len(cols)}"
    return f"{meta}\n\n列与类型:\n{dtypes_md}\n\n数据预览(前{min(max_rows, len(df))}行):\n{preview_md}"


def plot_chart_headless(output_manager: OutputManager):
    """
    返回与项目中签名一致的 plot_chart 函数，但不依赖 streamlit，转为落盘文件。
    """
    def _plot_chart(
        data: pd.DataFrame,
        chart_type: str,
        x_column: str,
        y_columns: Optional[List[str]] = None,
        legend_title: Optional[str] = None,
        title: str = "Chart",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ):
        if chart_type not in ["bar", "line", "pie", "scatter"]:
            print(f"[WARN] chart_type '{chart_type}' not supported.", file=sys.stderr)
            return

        if chart_type == "pie":
            # y_columns[0] 作为 values
            fig = px.pie(
                data,
                names=x_column,
                values=y_columns[0] if y_columns else None,
                title=title,
                color_discrete_sequence=colors,
            )
        else:
            fig = go.Figure()
            series = y_columns or []
            for i, y_col in enumerate(series):
                if chart_type == "bar":
                    fig.add_trace(
                        go.Bar(
                            x=data[x_column],
                            y=data[y_col],
                            name=y_col,
                            marker_color=(colors[i] if colors and i < len(colors) else None),
                        )
                    )
                elif chart_type == "line":
                    fig.add_trace(
                        go.Scatter(
                            x=data[x_column],
                            y=data[y_col],
                            mode="lines",
                            name=y_col,
                            line=dict(color=(colors[i] if colors and i < len(colors) else None)),
                        )
                    )
                elif chart_type == "scatter":
                    fig.add_trace(
                        go.Scatter(
                            x=data[x_column],
                            y=data[y_col],
                            mode="markers",
                            name=y_col,
                            marker=dict(color=(colors[i] if colors and i < len(colors) else None)),
                        )
                    )

            fig.update_layout(
                title=title,
                xaxis_title=(xlabel if xlabel else x_column),
                yaxis_title=(ylabel if ylabel else ("Value" if y_columns else "")),
                legend_title=legend_title,
            )

        # 保存文件
        out_path = output_manager.next_path()
        _, ext = os.path.splitext(out_path)
        ext = ext.lower()
        try:
            if ext == ".html":
                fig.write_html(
                    out_path,
                    include_plotlyjs="cdn",
                    full_html=True,
                    config={"displaylogo": False}
                )
            else:
                # 需要 kaleido 支持保存为静态图片
                fig.write_image(out_path)  # 若未安装 kaleido 会抛错
        except Exception as e:
            # 回退保存为 html
            fallback = os.path.splitext(out_path)[0] + ".html"
            fig.write_html(
                fallback,
                include_plotlyjs="cdn",
                full_html=True,
                config={"displaylogo": False}
            )
            print(f"[WARN] 保存为 {ext} 失败，已回退为 HTML: {fallback} ({e})", file=sys.stderr)
        else:
            print(f"[OK] 已保存图表到: {out_path}")

    return _plot_chart


def link_llm2_cli(text: str, df: Optional[pd.DataFrame], output_manager: OutputManager):
    """
    解析 LLM 输出：
      1) JSON 中的 'def_name' 列表，执行诸如 plot_chart(...) 或 write_df_to_excel(...)
      2) ```python 代码块，进行简要安全检查后执行（支持 df 作为参数）
    """
    # 兼容包含 DataFrame 花括号的 JSON
    pattern_json = r'\{[^{}]*\{.*?\}[^{}]*\}'
    pattern_py = r"(?<=```python\n)(.*?)(?=\n```)"

    matches_json = re.findall(pattern_json, text)
    matches_py = re.findall(pattern_py, text, re.DOTALL)

    # 注入执行上下文（仅提供必要符号）
    exec_globals = {
        "pd": pd,
        "plot_chart": plot_chart_headless(output_manager),
        "write_df_to_excel": write_df_to_excel,
        # 可按需加入更多白名单函数
    }

    if matches_json:
        for block in matches_json:
            try:
                payload = json.loads(block)
                defs = payload.get("def_name", [])
            except Exception:
                print("[WARN] JSON 解析失败，跳过该片段。", file=sys.stderr)
                continue
            for call in defs:
                try:
                    # 在受限上下文中执行
                    exec(call, exec_globals, None)
                except Exception as e:
                    print(f"[WARN] 指令执行失败: {call} ({e})", file=sys.stderr)

    if matches_py:
        for code in matches_py:
            try:
                if not check_code_safety(code):
                    print("[ERROR] 代码包含不允许的库，已终止执行。", file=sys.stderr)
                    continue
                # 允许以 df 作为参数运行
                _ = run_generated_code(code, df=df)
            except Exception as e:
                print(f"[WARN] 代码执行失败: {e}", file=sys.stderr)

    # 若两者都没命中，则原样返回文本供参考
    if not matches_json and not matches_py:
        return text
    return None


def main():
    parser = argparse.ArgumentParser(description="从 Excel + 提示词生成图表（命令行版）")
    parser.add_argument("--prompt", required=True, help="提示词（如：将文件数据进行画图）")
    parser.add_argument("--file", required=True, help="上传的 Excel 文件路径（.xlsx）")
    parser.add_argument("--output", required=True, help="输出文件路径（.html 或 .png 等，支持多图自动加序号）")
    parser.add_argument("--model", default="deepseek-v3-aliyun", help="LLM 模型名称，默认 deepseek-v3-aliyun")
    # 兼容别名 --mode
    parser.add_argument("--mode", dest="model", help="等同于 --model，兼容别名")
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("MINDCRAFT_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ),
        help="API Key（支持环境变量：MINDCRAFT_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY）",
    )
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI 兼容接口 base_url（默认 https://api.deepseek.com）")
    # 控制发给 LLM 的数据体量，避免超长上下文
    parser.add_argument("--max-preview-rows", type=int, default=60, help="预览excel数据行数上限（默认 60）")
    parser.add_argument("--max-preview-cols", type=int, default=12, help="预览excel数据列数上限（默认 12）")
    parser.add_argument("--columns", default=None, help="预览列白名单（逗号分隔，如: time,LastPrice）")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[ERROR] 文件不存在: {args.file}", file=sys.stderr)
        sys.exit(2)
    if not args.api_key:
        print("[ERROR] 缺少 API Key，请通过 --api-key 或环境变量 MINDCRAFT_API_KEY 提供。", file=sys.stderr)
        sys.exit(2)

    # 读取数据
    try:
        df = pd.read_excel(args.file)
    except Exception as e:
        print(f"[ERROR] 读取 Excel 失败: {e}", file=sys.stderr)
        sys.exit(2)

    # 组装对话内容（与 UI 逻辑一致：表格转 markdown 提示）
    prompt = args.prompt
    preview_md = build_df_preview_markdown(
        df,
        max_rows=args.max_preview_rows,
        max_cols=args.max_preview_cols,
        select_columns=args.columns,
    )
    text_2 = f"{prompt}\n请根据下述数据预览进行分析与作图（注意：仅展示部分行列，勿假设未展示数据）。\n{preview_md}"

    # 调用 LLM
    try:
        resp = llm_model2(text_2, model=args.model, API_key=args.api_key, base_url=args.base_url)
        text_out = llm_text2(resp)
        print("\n[LLM OUTPUT]\n" + text_out)
    except Exception as e:
        print(f"[ERROR] 模型调用失败: {e}", file=sys.stderr)
        sys.exit(2)

    # 解析并执行绘图/保存指令
    output_manager = OutputManager(args.output)
    result = link_llm2_cli(text_out, df=df, output_manager=output_manager)
    if isinstance(result, str) and result.strip():
        # 未解析出可执行指令时，将原始文本保存为 .txt 旁路输出
        base, _ = os.path.splitext(args.output)
        log_path = f"{base}_llm.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"[INFO] 无可执行指令，已将 LLM 文本结果保存到: {log_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()


