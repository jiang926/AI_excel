# AI_excel
一个最小可用的“读取 Excel + 调用 LLM 生成绘图指令 + 输出 Plotly 图表”的命令行工具。示例输出见 `out_file.html`。

## 1. 安装
```bash
python3.12 -m pip install -r requirements.txt
```

## 2. 运行
查看参数：
```bash
python3.12 cli_plot.py --help
```

示例（DeepSeek）：
```bash
export DEEPSEEK_API_KEY="你的_DeepSeek_API_KEY"
python3.12 cli_plot.py \
  --prompt "分析每个地区的总人口，且学历各占比重是多少, 在一个图中，使用柱状图，来分析学历比重，图中要添加中文描述" \
  --file "地区学历统计.xlsx" \
  --output "out_file.html" \
  --mode "deepseek-chat"
  --also-image --image-format png --image-scale 2
```

常用参数：
- `--columns "Time,LastPrice"`：只发送指定列，降低 token
- `--offline`：HTML 内联 Plotly，离线也可查看
- `--hide-modebar`：隐藏交互工具栏
- `--also-image`：生成 HTML 后自动额外导出图片（与 HTML 同名）
- `--image-format png|svg|jpeg|webp`：图片格式（默认 png）
- `--image-scale`、`--image-width`、`--image-height`：清晰度或尺寸控制（需要 kaleido）

## 3. 容器运行
构建：
```bash
docker build -t ai-excel:latest .
```
运行（挂载当前项目目录，传入 API Key）：
```bash
docker run --rm -it \
  -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
  -v /mnt/sde/AI_excel:/app \
  ai-excel:latest python cli_plot.py --help
```

## 4. 说明
- 默认 `BASE_URL` 为 `https://api.deepseek.com`，可通过环境变量覆盖。
- 读取 Excel 时会根据扩展名自动选择引擎（`.xlsx` 使用 `openpyxl`，`.xls` 使用 `xlrd`）。
- 导出 PNG 需要安装 Chrome/Chromium（供 `kaleido` 调用）；否则建议导出 HTML。
  - 若导出失败，可执行 `plotly_get_chrome` 安装浏览器后重试。