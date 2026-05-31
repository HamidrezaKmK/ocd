"""Personalized configuration: spending categories, merchant memory, and run metadata.

The user defines categories once (name + description + monthly limit); the description is
fed to the classifier so categorization reflects *their* mental model. Corrections made
during review accumulate in a merchant-memory map that makes future runs deterministic and
progressively more accurate.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from . import paths


class Category(BaseModel):
    """A user-defined spending category."""

    name: str
    description: str = ""
    monthly_limit: float = Field(0.0, ge=0.0)
    # Hidden categories are still classifiable (e.g. card payments / transfers between your own
    # accounts) but are excluded from the report so they don't double-count as spending.
    hidden: bool = False

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("category name cannot be empty")
        return v


# The catch-all bucket. Always available so the classifier can fall back to it and the
# review step can surface anything the model was unsure about.
UNCATEGORIZED = "Uncategorized"

DEFAULT_CATEGORIES = [
    Category(name="Groceries", description="Supermarkets, grocery stores, food markets", monthly_limit=600),
    Category(name="Dining", description="Restaurants, cafes, bars, food delivery, coffee shops", monthly_limit=300),
    Category(name="Transport", description="Gas, fuel, rideshare, transit, parking, tolls", monthly_limit=200),
    Category(name="Shopping", description="Retail, clothing, electronics, online marketplaces", monthly_limit=300),
    Category(name="Utilities & Bills", description="Phone, internet, electricity, water, insurance", monthly_limit=400),
    Category(name="Entertainment", description="Streaming, movies, games, events, subscriptions", monthly_limit=150),
    Category(name="Health", description="Pharmacy, doctors, gym, fitness, wellness", monthly_limit=200),
    Category(name="Travel", description="Flights, hotels, car rental, vacation spending", monthly_limit=300),
]


class CategoryConfig(BaseModel):
    """The full set of categories plus convenience accessors."""

    categories: list[Category]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.categories]

    @property
    def all_names(self) -> list[str]:
        """Category names plus the always-available Uncategorized bucket."""
        return self.names + [UNCATEGORIZED]

    @property
    def hidden_names(self) -> set[str]:
        """Categories marked hidden — excluded from the report (e.g. payments/transfers)."""
        return {c.name for c in self.categories if c.hidden}

    def limit_for(self, name: str) -> float:
        for c in self.categories:
            if c.name == name:
                return c.monthly_limit
        return 0.0

    def get(self, name: str) -> Optional[Category]:
        for c in self.categories:
            if c.name == name:
                return c
        return None


# --------------------------------------------------------------------------- #
# Categories I/O
# --------------------------------------------------------------------------- #
def categories_exist() -> bool:
    return paths.CATEGORIES_YAML.exists()


def load_categories() -> CategoryConfig:
    """Load categories from YAML, or return defaults if none configured yet."""
    if not paths.CATEGORIES_YAML.exists():
        return CategoryConfig(categories=list(DEFAULT_CATEGORIES))
    data = yaml.safe_load(paths.CATEGORIES_YAML.read_text()) or {}
    raw = data.get("categories", [])
    cats = [Category(**c) for c in raw]
    if not cats:
        cats = list(DEFAULT_CATEGORIES)
    return CategoryConfig(categories=cats)


def save_categories(config: CategoryConfig) -> None:
    paths.ensure_dirs()
    payload = {
        "categories": [
            {"name": c.name, "description": c.description,
             "monthly_limit": c.monthly_limit, "hidden": c.hidden}
            for c in config.categories
        ]
    }
    paths.CATEGORIES_YAML.write_text(yaml.safe_dump(payload, sort_keys=False))


# --------------------------------------------------------------------------- #
# Merchant memory: normalized merchant -> category, learned from user corrections
# --------------------------------------------------------------------------- #
def load_merchant_memory() -> dict[str, str]:
    if not paths.MERCHANT_MEMORY_YAML.exists():
        return {}
    data = yaml.safe_load(paths.MERCHANT_MEMORY_YAML.read_text()) or {}
    return dict(data.get("merchants", {}))


def save_merchant_memory(memory: dict[str, str]) -> None:
    paths.ensure_dirs()
    payload = {"merchants": dict(sorted(memory.items()))}
    paths.MERCHANT_MEMORY_YAML.write_text(yaml.safe_dump(payload, sort_keys=False))


def remember_merchants(corrections: dict[str, str]) -> dict[str, str]:
    """Merge new merchant->category corrections into the persisted memory."""
    memory = load_merchant_memory()
    memory.update({k: v for k, v in corrections.items() if k and v})
    save_merchant_memory(memory)
    return memory


# --------------------------------------------------------------------------- #
# Run metadata: the finalization gate between Step 2 and Step 3
# --------------------------------------------------------------------------- #
class RunMeta(BaseModel):
    finalized: bool = False
    finalized_at: Optional[str] = None
    period: Optional[str] = None
    n_transactions: int = 0


def load_meta() -> RunMeta:
    if not paths.CATEGORIZED_META.exists():
        return RunMeta()
    data = yaml.safe_load(paths.CATEGORIZED_META.read_text()) or {}
    return RunMeta(**data)


def save_meta(meta: RunMeta) -> None:
    paths.ensure_dirs()
    paths.CATEGORIZED_META.write_text(yaml.safe_dump(meta.model_dump(), sort_keys=False))


def mark_finalized(period: Optional[str] = None, n_transactions: int = 0) -> RunMeta:
    meta = RunMeta(
        finalized=True,
        finalized_at=date.today().isoformat(),
        period=period,
        n_transactions=n_transactions,
    )
    save_meta(meta)
    return meta


def mark_draft(n_transactions: int = 0) -> RunMeta:
    meta = RunMeta(finalized=False, n_transactions=n_transactions)
    save_meta(meta)
    return meta
