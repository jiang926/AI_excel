import os


def get_base_url() -> str:
    """
    获取默认的 LLM 接口地址，优先读取环境变量 BASE_URL
    """
    return os.environ.get("BASE_URL", "https://api.deepseek.com")


def get_api_key() -> str | None:
    """
    获取默认的 API Key，按优先级读取：
    MINDCRAFT_API_KEY > DEEPSEEK_API_KEY > OPENAI_API_KEY
    """
    return (
        os.environ.get("MINDCRAFT_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


