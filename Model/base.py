from dataclasses import dataclass


@dataclass(kw_only=True)
class BaseModel:
    """Parent for all models; supplies the shared id."""
    id: int | None = None
