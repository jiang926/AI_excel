# AI_excel
这是一个可以自动操作Excel的AI工具

软件效果视频：[只说人话就可以做表格了！我造了一个自动操作Excel表的网站](https://www.bilibili.com/video/BV1JzKpeTEkF/?vd_source=28ba27f4f650db659b1dd1ace9f5fc5c)

本项目调用大语言模型的API接口，并让模型能够调用操作Excel表文件的函数

项目用streamlit编写界面


运行命令 `pip install -r requirements.txt` 即可安装所有所需依赖。

python3.12 cli_plot.py --help  可查看参数说明

demo:
    python3.12 cli_plot.py   --prompt "分析每个地区的总人口，且学历各占比重是多少, 在一个图中，使用柱状图，来分析学历比重，图中要添加中文描述"  --file "地区学历统计.xlsx"   --output "out_file.html"   --mode "deepseek-chat"