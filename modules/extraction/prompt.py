from pathlib import Path
from jinja2 import Environment, FileSystemLoader,StrictUndefined

# 动态计算模板目录
PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

env = Environment(
    loader=FileSystemLoader(PROMPT_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True)

def render(name: str, **kwargs) -> str:
    return env.get_template(f"{name}.j2").render(**kwargs)

