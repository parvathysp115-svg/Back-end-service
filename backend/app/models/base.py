import json
import uuid
from datetime import datetime
from typing import Any, List, Optional
from sqlalchemy import DateTime, String, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from pgvector.sqlalchemy import Vector
from app.core.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class PortableVector(TypeDecorator):
    """
    Custom TypeDecorator that seamlessly maps to pgvector Vector(dim)
    on PostgreSQL and to JSON/text on other engines like SQLite for tests.
    """
    impl = Vector
    cache_ok = True

    def __init__(self, dim: int = 768, *args: Any, **kwargs: Any):
        super().__init__(dim, *args, **kwargs)
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if dialect.name == "postgresql":
                return [float(x) for x in value]
            return list(value)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                # Handle stringified vector from sqlite or json
                parsed = json.loads(value)
                return [float(x) for x in parsed]
            except Exception:
                return value
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
        return value


class TimestampMixin:
    created_at = DateTime(timezone=True)
    updated_at = DateTime(timezone=True)
