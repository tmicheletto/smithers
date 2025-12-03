"""Server entry point for the chatbot API."""

import uvicorn

from .config import settings

if __name__ == "__main__":
    uvicorn.run(
        "smithers.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,  # Enable auto-reload for development
        log_level="info",
    )
