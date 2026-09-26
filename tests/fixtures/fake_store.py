"""A tiny in-memory stand-in for a pymongo collection, for tests only.

Why this exists: `pytest` has to pass on the MSI with no Atlas reachable, and the Core
Principle property tests (append-only, diary-before-belief, sightings-beat-inferences,
never-merge-when-unsure) are about the memory layer's behaviour, not about MongoDB's. It is
not a Mongo emulator — `mongomock` supports neither time series collections nor
`$vectorSearch`, which is why neither is used.

Clearly a fixture, lives under tests/fixtures/, and is never imported by production code.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Iterable


def _get(doc: dict, path: str) -> Any:
    """Dotted lookup: "meta.session" -> doc["meta"]["session"]."""
    current: Any = doc
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _matches(doc: dict, query: dict) -> bool:
    for field, expected in query.items():
        actual = _get(doc, field)
        if isinstance(expected, dict):
            for op, operand in expected.items():
                if op == "$eq" and actual != operand:
                    return False
                if op == "$ne" and actual == operand:
                    return False
                if op == "$in" and actual not in operand:
                    return False
                if op == "$nin" and actual in operand:
                    return False
                if op == "$gte" and not (actual is not None and actual >= operand):
                    return False
                if op == "$lte" and not (actual is not None and actual <= operand):
                    return False
                if op == "$gt" and not (actual is not None and actual > operand):
                    return False
                if op == "$lt" and not (actual is not None and actual < operand):
                    return False
                if op == "$exists" and (actual is not None) != bool(operand):
                    return False
                if op == "$regex":
                    import re
                    if actual is None or not re.search(operand, str(actual), re.I):
                        return False
        elif actual != expected:
            return False
    return True


class FakeCursor:
    def __init__(self, docs: list[dict]):
        self._docs = docs

    def sort(self, field: str, direction: int = 1) -> "FakeCursor":
        self._docs.sort(key=lambda d: (_get(d, field) is None, _get(d, field)),
                        reverse=direction < 0)
        return self

    def limit(self, n: int) -> "FakeCursor":
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(copy.deepcopy(self._docs))

    def __len__(self) -> int:
        return len(self._docs)


class FakeResult:
    def __init__(self, inserted_id=None, matched_count=0, modified_count=0, upserted_id=None):
        self.inserted_id = inserted_id
        self.matched_count = matched_count
        self.modified_count = modified_count
        self.upserted_id = upserted_id


class FakeCollection:
    """Supports only the operations the memory layer actually uses."""

    def __init__(self, name: str = "fake"):
        self.name = name
        self.docs: list[dict] = []
        self._ids = itertools.count(1)
        #: Every write, in order. Tests assert on this to prove append-only behaviour.
        self.writes: list[tuple[str, dict]] = []

    # ------------------------------------------------------------------------- writes

    def insert_one(self, doc: dict) -> FakeResult:
        stored = copy.deepcopy(doc)
        stored.setdefault("_id", doc.get("_id", next(self._ids)))
        self.docs.append(stored)
        doc["_id"] = stored["_id"]
        self.writes.append(("insert", copy.deepcopy(stored)))
        return FakeResult(inserted_id=stored["_id"])

    def insert_many(self, docs: Iterable[dict]) -> FakeResult:
        for doc in docs:
            self.insert_one(doc)
        return FakeResult()

    def update_one(self, query: dict, update: dict, upsert: bool = False) -> FakeResult:
        for doc in self.docs:
            if _matches(doc, query):
                self._apply(doc, update)
                self.writes.append(("update", {"query": query, "update": copy.deepcopy(update)}))
                return FakeResult(matched_count=1, modified_count=1)
        if upsert:
            base = {k: v for k, v in query.items() if not isinstance(v, dict)}
            self._apply(base, update)
            result = self.insert_one(base)
            return FakeResult(upserted_id=result.inserted_id)
        return FakeResult()

    def replace_one(self, query: dict, replacement: dict, upsert: bool = False) -> FakeResult:
        for index, doc in enumerate(self.docs):
            if _matches(doc, query):
                keep_id = doc["_id"]
                self.docs[index] = {**copy.deepcopy(replacement), "_id": keep_id}
                self.writes.append(("replace", copy.deepcopy(replacement)))
                return FakeResult(matched_count=1, modified_count=1)
        if upsert:
            return self.insert_one(copy.deepcopy(replacement))
        return FakeResult()

    @staticmethod
    def _apply(doc: dict, update: dict) -> None:
        for op, fields in update.items():
            if op == "$set":
                for path, value in fields.items():
                    target, _, leaf = path.rpartition(".")
                    node = doc
                    for part in filter(None, target.split(".")):
                        node = node.setdefault(part, {})
                    node[leaf] = copy.deepcopy(value)
            elif op == "$push":
                for path, value in fields.items():
                    doc.setdefault(path, []).append(copy.deepcopy(value))
            elif op == "$inc":
                for path, value in fields.items():
                    doc[path] = doc.get(path, 0) + value
            elif op == "$setOnInsert":
                for path, value in fields.items():
                    doc.setdefault(path, copy.deepcopy(value))
            else:
                raise NotImplementedError(f"FakeCollection does not implement {op}")

    # -------------------------------------------------------------------------- reads

    def find(self, query: dict | None = None, projection=None) -> FakeCursor:
        return FakeCursor([copy.deepcopy(d) for d in self.docs if _matches(d, query or {})])

    def find_one(self, query: dict | None = None, projection=None) -> dict | None:
        for doc in self.docs:
            if _matches(doc, query or {}):
                return copy.deepcopy(doc)
        return None

    def count_documents(self, query: dict | None = None) -> int:
        return sum(1 for d in self.docs if _matches(d, query or {}))

    estimated_document_count = count_documents


class FakeDatabase:
    """Dict of FakeCollections, addressable like a pymongo Database."""

    def __init__(self):
        self._collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        if name not in self._collections:
            self._collections[name] = FakeCollection(name)
        return self._collections[name]

    def list_collection_names(self) -> list[str]:
        return sorted(self._collections)
