from typing import Annotated, Any, Dict
from typing_extensions import Literal
from pydantic import Field
from datetime import datetime
from configs.path_config import path_config
from schemas.base_schema import BaseSchema


class LogSchema(BaseSchema):
    sink: Annotated[
        str, Field(default_factory=lambda: f"{path_config.logs}/{datetime.now()}.log")
    ]  # Do not touch
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {thread.name} | {name}:{function}:{line} - {message}"
    )
    rotation: str = "10 MB"
    retention: str = "7 days"
    colorize: bool = True
