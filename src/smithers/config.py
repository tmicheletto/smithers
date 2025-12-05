"""Configuration for the agentic application."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file into environment variables
# This makes API keys available to PydanticAI providers
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Settings can be configured via:
    - Environment variables
    - .env file in the project root
    - Constructor arguments
    """

    # Model configuration (LLM identifier)
    # Align with LangChain `ChatOpenAI` model used in the chain.
    model: str = "gpt-4o-mini"

    # Logging configuration
    enable_logfire: bool = True

    # Agent configuration
    agent_retries: int = 2

    # API configuration
    api_timeout: int = 30
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    # RAG configuration
    class RAGSettings(BaseSettings):
        """Configuration for RAG system."""

        index_name: str = "smithers-knowledge-index"
        embedding_model: str = "text-embedding-3-small"
        search_top_k: int = 5
        dimension: int = 1536  # text-embedding-3-small output dimension

        class Config:
            """Pydantic settings configuration."""

            env_prefix = "RAG_"

    rag: RAGSettings = RAGSettings()
    data_dir: Path = Path(__file__).parent.parent.parent / "data"

    # API Keys (loaded from .env, made available in environment for providers)
    openai_api_key: str | None = None

    class Config:
        """Pydantic settings configuration."""

        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()
