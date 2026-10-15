
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ---------------- 模型服务 ----------------
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str
    deepseek_model: str = "deepseek-flash"

    # ---------------- 文档处理（D1 用）----------------
    # 切分粒度。500 是中文文档的常用起点，D1 你要自己调出最合适的值
    chunk_size: int = 500
    # ★ D1 实测调出来的值：50 会失效，100 才稳。
    #   原因：LangChain 的重叠是【按切分单元整句留】的，不是按字符。
    #   中文一句话 60–90 字，overlap=50 时一句话都留不下 → 重叠归零。
    #   提到 100（= chunk_size 的 20%）后，三份文档的相邻重叠稳定在 52–98 字，不再归零。
    chunk_overlap: int = 100
    # 上传的原始文件放哪（已在 .gitignore 里排除）
    data_dir: Path = BASE_DIR / "data"

    # ---------------- 检索（D2–D5 用）----------------
    # Embedding 模型。中文场景【必须】用中文模型，别用 all-MiniLM 这类英文模型
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # 向量库落盘位置
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    # 召回条数 / 精排后保留条数
    recall_top_k: int = 20
    rerank_top_k: int = 5

    # ---------------- Agent（D6–D8 用）----------------
    # ★ Agent 循环的最大步数 —— 防死循环烧钱，必须有
    max_agent_steps: int = 10

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")


settings = Settings()
