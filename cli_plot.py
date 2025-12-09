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
import logging
import os
import sys
import re
import json
from typing import Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gpt_data import llm_model2, llm_text2
import config


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
    # 截断每个单元格，避免超长字符串（避免 applymap 的 FutureWarning）
    preview = preview.astype(str).apply(
        lambda s: s.where(s.str.len() <= max_cell_chars, s.str.slice(0, max_cell_chars) + "…")
    )
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
        supported = ["bar", "line", "pie", "scatter", "heatmap"]
        if chart_type not in supported:
            print(f"[WARN] chart_type '{chart_type}' not supported.", file=sys.stderr)
            return

        # 统一的默认配色
        palette = colors or px.colors.qualitative.Plotly

        def _apply_layout(fig_obj: go.Figure):
            fig_obj.update_layout(
                title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20)),
                legend=dict(title=legend_title, orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                template="plotly_white",
                font=dict(family="Arial", size=13),
                margin=dict(l=60, r=30, t=70, b=60),
                colorway=palette,
            )
            fig_obj.update_xaxes(
                title=(xlabel if xlabel else x_column),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                zeroline=False,
            )
            fig_obj.update_yaxes(
                title=(ylabel if ylabel else ("Value" if y_columns else "")),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                zeroline=False,
            )
            # 线/散点交互更好
            if chart_type in ["line", "scatter"]:
                fig_obj.update_layout(hovermode="x unified")

        if chart_type == "pie":
            # y_columns[0] 作为 values
            fig = px.pie(
                data,
                names=x_column,
                values=y_columns[0] if y_columns else None,
                title=title,
                color_discrete_sequence=palette,
                hole=0.25,
            )
            fig.update_traces(textinfo="percent+label", textposition="inside")
            _apply_layout(fig)
        elif chart_type == "heatmap":
            # 适配简单热力图：x 轴为 x_column，z 为首个 y_columns 数值列
            if not y_columns:
                print("[WARN] heatmap 需要至少一个数值列，已跳过。", file=sys.stderr)
                return
            z_vals = [data[y_columns[0]].tolist()]
            fig = go.Figure(
                data=go.Heatmap(
                    z=z_vals,
                    x=data[x_column],
                    y=[legend_title or y_columns[0]],
                    colorscale="Blues",
                    colorbar_title=legend_title or "value",
                )
            )
            _apply_layout(fig)
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
                            marker_color=(palette[i] if i < len(palette) else None),
                            marker_line_width=0.5,
                        )
                    )
                elif chart_type == "line":
                    fig.add_trace(
                        go.Scatter(
                            x=data[x_column],
                            y=data[y_col],
                            mode="lines",
                            name=y_col,
                            line=dict(
                                color=(palette[i] if i < len(palette) else None),
                                width=2.2,
                                shape="spline",
                            ),
                        )
                    )
                elif chart_type == "scatter":
                    fig.add_trace(
                        go.Scatter(
                            x=data[x_column],
                            y=data[y_col],
                            mode="markers",
                            name=y_col,
                            marker=dict(
                                color=(palette[i] if i < len(palette) else None),
                                size=8,
                                opacity=0.9,
                            ),
                        )
                    )

            fig.update_layout(
                legend_title=legend_title,
            )
            _apply_layout(fig)

        # 保存文件
        out_path = output_manager.next_path()
        _, ext = os.path.splitext(out_path)
        ext = ext.lower()
        # 配置下载图片按钮（摄像机图标）
        filename_for_image = os.path.splitext(os.path.basename(out_path))[0]
        cfg = {
            "displaylogo": False,
            "displayModeBar": not getattr(output_manager, "_hide_modebar", False),
            "toImageButtonOptions": {
                "format": "png",          # png/svg/jpeg/webp
                "filename": filename_for_image,
                "scale": 2,               # 提高清晰度
                # 也可设置固定宽高: "width": 1200, "height": 700
            },
        }
        try:
            if ext == ".html":
                fig.write_html(
                    out_path,
                    include_plotlyjs=("cdn" if not getattr(output_manager, "_offline", False) else True),
                    full_html=True,
                    config=cfg,
                )
                # 额外导出图片
                if getattr(output_manager, "_also_image", False):
                    fmt = getattr(output_manager, "_image_format", "png")
                    scale = getattr(output_manager, "_image_scale", 2)
                    width = getattr(output_manager, "_image_width", None)
                    height = getattr(output_manager, "_image_height", None)
                    image_out = os.path.splitext(out_path)[0] + f".{fmt}"
                    try:
                        fig.write_image(
                            image_out,
                            format=fmt,
                            scale=scale,
                            width=width,
                            height=height,
                        )
                        print(f"[OK] 额外导出图片: {image_out}")
                    except Exception as ie:
                        print(
                            f"[WARN] 图片导出失败，已保留 HTML。可能需要安装 Chrome/Chromium（kaleido 依赖）。"
                            f" 可尝试运行: plotly_get_chrome ({ie})",
                            file=sys.stderr,
                        )
            else:
                # 需要 kaleido 支持保存为静态图片
                fig.write_image(out_path)  # 若未安装 kaleido 会抛错
        except Exception as e:
            # 回退保存为 html
            fallback = os.path.splitext(out_path)[0] + ".html"
            fig.write_html(
                fallback,
                include_plotlyjs=("cdn" if not getattr(output_manager, "_offline", False) else True),
                full_html=True,
                config=cfg,
            )
            print(f"[WARN] 保存为 {ext} 失败，已回退为 HTML: {fallback} ({e})", file=sys.stderr)
        else:
            print(f"[OK] 已保存图表到: {out_path}")

    return _plot_chart


def link_llm2_cli(text: str, df: Optional[pd.DataFrame], output_manager: OutputManager):
    """
    解析 LLM 输出：仅允许 JSON 里的 'def_name' 指令，执行 plot_chart(...)
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
        # 仅暴露绘图函数，禁止写文件/执行任意代码
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

    # 出于安全与简化，仅支持 JSON 指令；忽略 python 代码块

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
    parser.add_argument("--base-url", default=config.get_base_url(), help="OpenAI 兼容接口 base_url（默认使用环境变量 BASE_URL 或 https://api.deepseek.com）")
    # 控制发给 LLM 的数据体量，避免超长上下文
    parser.add_argument("--max-preview-rows", type=int, default=60, help="预览excel数据行数上限（默认 60）")
    parser.add_argument("--max-preview-cols", type=int, default=12, help="预览excel数据列数上限（默认 12）")
    parser.add_argument("--columns", default=None, help="预览列白名单（逗号分隔，如: time,LastPrice）")
    # 输出控制
    parser.add_argument("--offline", action="store_true", help="导出 HTML 时内联 Plotly 脚本（离线可用）")
    parser.add_argument("--hide-modebar", action="store_true", help="隐藏交互工具栏")
    # 自动导出图片
    parser.add_argument("--also-image", action="store_true", help="在生成 HTML 后，同时导出图片文件")
    parser.add_argument("--image-format", default="png", choices=["png", "svg", "jpeg", "webp"], help="图片格式（默认 png）")
    parser.add_argument("--image-scale", type=int, default=2, help="图片缩放倍数，提高清晰度（默认 2）")
    parser.add_argument("--image-width", type=int, default=None, help="图片宽度（可选）")
    parser.add_argument("--image-height", type=int, default=None, help="图片高度（可选）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not os.path.exists(args.file):
        print(f"[ERROR] 文件不存在: {args.file}", file=sys.stderr)
        sys.exit(2)
    api_key = args.api_key or config.get_api_key()
    if not api_key:
        print("[ERROR] 缺少 API Key，请通过 --api-key 或环境变量提供。", file=sys.stderr)
        sys.exit(2)

    # 读取数据
    try:
        _, ext = os.path.splitext(args.file.lower())
        if ext == ".xls":
            df = pd.read_excel(args.file, engine="xlrd")
        elif ext == ".xlsx":
            df = pd.read_excel(args.file, engine="openpyxl")
        else:
            # 尝试常见引擎
            try:
                df = pd.read_excel(args.file, engine="openpyxl")
            except Exception:
                df = pd.read_excel(args.file, engine="xlrd")
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
        resp = llm_model2(text_2, model=args.model, API_key=api_key, base_url=args.base_url)
        text_out = llm_text2(resp)
        print("\n[LLM OUTPUT]\n" + text_out)
    except Exception as e:
        print(f"[ERROR] 模型调用失败: {e}", file=sys.stderr)
        sys.exit(2)

    # 若模型直接返回 HTML，则按指定路径落盘
    html_match = re.search(r"<!doctype html|<html", text_out, re.IGNORECASE)
    if html_match:
        html_start = html_match.start()
        html_body = text_out[html_start:]
        base, ext = os.path.splitext(args.output)
        out_path = args.output if ext else f"{args.output}.html"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_body)
        print(f"[OK] 已直接保存模型返回的 HTML 到: {out_path}")
        sys.exit(0)

    # 解析并执行绘图/保存指令（非 HTML 返回）
    output_manager = OutputManager(args.output)
    # 传递导出选项
    setattr(output_manager, "_offline", bool(args.offline))
    setattr(output_manager, "_hide_modebar", bool(args.hide_modebar))
    setattr(output_manager, "_also_image", bool(args.also_image))
    setattr(output_manager, "_image_format", args.image_format)
    setattr(output_manager, "_image_scale", args.image_scale)
    setattr(output_manager, "_image_width", args.image_width)
    setattr(output_manager, "_image_height", args.image_height)
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


