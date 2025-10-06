from typing import Any, Dict, Mapping
from pydantic import BaseModel, Field, model_validator


class BaseSchema(BaseModel, Mapping[str, Any]):
    """
    A Pydantic BaseModel that:
      - Collects any undeclared fields into `extra: Dict[str, Any]`
      - Exposes a Mapping interface (so you can do dict-style access/iteration)
    """

    extra: Dict[str, Any] = Field(default_factory=dict)

    # Allow arbitrary extra fields, and re-validate on assignment
    model_config = {"extra": "allow", "validate_assignment": True}

    @model_validator(mode="before")
    @classmethod
    def collect_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        declared = set(cls.model_fields.keys())
        extras = {k: v for k, v in values.items() if k not in declared}
        values["extra"] = extras
        return values

    def to_dict(self) -> Dict[str, Any]:
        base = self.model_dump(exclude={"extra"}, mode="python")
        return {**base, **self.extra}

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()})"
