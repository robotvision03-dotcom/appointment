"""Run the FastAPI app: python -m src or uvicorn src.main:app"""

from src.config import config
from src.main import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())
