"""Configuration for the agentic application."""

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

    # Model configuration
    model: str = "gemini-2.5-flash-lite"

    # Logging configuration
    enable_logfire: bool = True

    # Agent configuration
    agent_retries: int = 2

    # API configuration
    api_timeout: int = 30
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]  # Configure for production

    # API Keys (loaded from .env, made available in environment for providers)
    google_api_key: str | None = None

    class Config:
        """Pydantic settings configuration."""

        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
