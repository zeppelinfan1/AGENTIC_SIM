from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    catalog: str = "main_catalog"
