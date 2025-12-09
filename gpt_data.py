# -*- coding: utf-8 -*-
from openai import OpenAI
import json
import re
import base64

file_path = 'C:/files/python_files/gpt_excel/入库单.xlsx'

# 本地模型名称
model_local = "deepseek-r1:1.5b"



def get_image_base64(image_file):
    """Convert an image file to a base64 string."""
    if image_file is not None:
        encoded_string = base64.b64encode(image_file.read()).decode()  # 编码并转换为可读字符串
        return encoded_string
    return None


def llm_model2(content, model=None, API_key=None, image_file=None, base_url=None):
    if model is None:
        model = "deepseek-v3-volcengine"
    else:
        model = model

    if model == "本地模型":
        base_url_final = 'http://localhost:11434/v1/'
        api_key = 'ollama'
        model = model_local
    else:
        # 优先使用传入的 base_url，其次保留默认为 DeepSeek
        base_url_final = base_url if base_url else 'https://api.deepseek.com'
        api_key = API_key
    client = OpenAI(base_url=base_url_final, api_key=api_key)

    if image_file is not None:
        params = {
            "model": "Doubao-1.5-vision-pro-32k",
            "message": [
                {
                    "role": "user",
                    "content": [
                        # 使用 base64 编码传输
                        {
                            'type': 'image',
                            'source': {
                                'data': get_image_base64(image_file)
                            },
                        },
                        {
                            'type': 'text',
                            'text': content,
                        },
                    ]
                }

            ],
            "temperature": 0,
            "max_tokens": 8000,
            "stream": True
        }

    else:
        params = {
            "model": model,
            "message": [
                {
                    "role": "user",
                    "content": (
                        "分三步输出："
                        "1) 需求分析：简述用户想要的图表/指标和字段映射。"
                        "2) 数据处理计划：指出需要用到的列、聚合/分组方式、计算逻辑。"
                        "3) 生成最终 HTML：直接给出完整可用的 HTML（含 <html>），内嵌所需脚本/样式与所有图表。"
                        "注意：步骤1/2 请输出纯文本，不要写入 HTML 标签中；最终 HTML 单独从 <html> 开始。"
                        "不要返回 Markdown 代码块或额外解释；多图请放同一 HTML。"
                        f" 用户请求：{content}"
                    )
                }

            ],
            "temperature": 0,
            "max_tokens": 8000,
            "stream": True,
            # 启用思考模式（DeepSeek reasoning）
            # "extra_body": {"thinking": {"type": "enabled"}},
        }


    response = client.chat.completions.create(
        model=params.get("model"),
        messages=params.get("message"),
        temperature=params.get("temperature"),
        max_tokens=params.get("max_tokens"),
        stream=params.get("stream"),
        extra_body=params.get("extra_body"),
    )
    return response


def llm_text2(response):
    text = ''
    for i in response:
        content = i.choices[0].delta.content
        # content 可能为 None（如仅返回 usage），需跳过避免拼接错误
        if content is None:
            if getattr(i, "usage", None):
                print('\n请求花销usage:', i.usage)
            continue
        print(content, end='', flush=True)
        text += content
        #text_to_speech(content)
    else:
        print()
    return text


# 连接其他函数
def link_llm2(text):
    # 使用正则表达式查找{'def_name'
    # 正则表达式
    pattern = r'\{[^{}]*\}'

    # 使用正则表达式匹配
    match = re.findall(pattern, text)
    print(match)

    if match:
        for i_n in match:
            print(i_n)
            try:
                # 解析JSON数据
                json_data = json.loads(i_n)
                datas = json_data['def_name']
            except:
                print("解析JSON出错")
            # 执行函数
            for data in datas:
                try:
                    print(data)
                    exec(data)
                except:
                    str_text = "不能执行此动作"
                    print(str_text)
                    return str_text
    else:
        return text


def AI_run2(content, model, API_key):
    response = llm_model2(content, model, API_key)
    text = llm_text2(response)
    return text


if __name__ == '__main__':
    try:
        while True:
            content = input("写入需求:")
            text = AI_run2(content, model=None, API_key='1')
            #link_llm2(text)
    except KeyboardInterrupt:
        print("程序出错已退出。")