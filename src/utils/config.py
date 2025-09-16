import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

    default_model: str = Field("gpt-4o-mini", env="DEFAULT_MODEL")
    anthropic_model: str = Field("claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")

    data_dir: str = Field("./data", env="DATA_DIR")
    output_dir: str = Field("./outputs", env="OUTPUT_DIR")

    max_csv_rows: int = Field(10000, env="MAX_CSV_ROWS")
    chart_width: int = Field(800, env="CHART_WIDTH")
    chart_height: int = Field(600, env="CHART_HEIGHT")
    excel_theme: str = Field("default", env="EXCEL_THEME")

    class Config:
        env_file = ".env"
        case_sensitive = False

def get_settings() -> Settings:
    return Settings()

settings = get_settings()