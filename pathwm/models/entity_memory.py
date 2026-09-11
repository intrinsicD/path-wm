"""Single-writer entity records. Recognition remains learned; IDs are bookkeeping."""

import copy
import hashlib
import math

import torch

from pathwm.io import digest


class EntityMemory:
    """Bounded CPU store; snapshots are restart points, not concurrent transactions.

    Restoring an old snapshot rolls back the clock too. The caller owns durable,
    atomic snapshot publication and must select the latest committed snapshot.
    """

    def __init__(self, model, capacity=3, threshold=0.75, receipt_limit=64):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be positive")
        if type(receipt_limit) is not int or receipt_limit < 1:
            raise ValueError("receipt_limit must be positive")
        if not 0.5 < threshold < 1:
            raise ValueError("threshold must exceed one half and be below one")
        self._model = copy.deepcopy(model).cpu().eval().requires_grad_(False)
        fingerprint = hashlib.sha256(type(model).__qualname__.encode())
        for name, value in self._model.state_dict().items():
            fingerprint.update(name.encode())
            fingerprint.update(str((value.shape, value.dtype)).encode())
            fingerprint.update(value.detach().contiguous().numpy().tobytes())
        self._state = dict(
            schema=1,
            model=fingerprint.hexdigest(),
            capacity=capacity,
            threshold=float(threshold),
            receipt_limit=receipt_limit,
            clock=None,
            records=[],
            receipts=[],
        )

    @staticmethod
    def _descriptor(value):
        value = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
        if value.shape != (8,) or not torch.isfinite(value).all():
            raise ValueError("Expected a finite descriptor of length eight")
        if not torch.isclose(value.norm(), value.new_tensor(1.0), atol=1e-5):
            raise ValueError("Descriptor must have unit norm")
        return value.tolist()

    def _match(self, descriptor):
        records = self._state["records"]
        index, confidence = len(records), 1.0
        if records:
            query = torch.tensor([descriptor])
            memory = torch.tensor([[r["descriptor"] for r in records]])
            with torch.inference_mode():
                logits = self._model.match(query, memory)
                if (
                    logits.shape != (1, len(records) + 1)
                    or not torch.isfinite(logits).all()
                ):
                    raise ValueError("Invalid matcher scores")
                probability, chosen = logits.softmax(-1).max(-1)
                index, confidence = chosen.item(), probability.item()
        return index, confidence

    def lookup(self, descriptor):
        """Read-only recognition; unknown queries never allocate records."""
        descriptor = self._descriptor(descriptor)
        index, confidence = self._match(descriptor)
        known = confidence > self._state["threshold"] and index < len(
            self._state["records"]
        )
        return dict(entity_id=index if known else None, confidence=confidence)

    def observe(self, event_id, descriptor, timestamp, *, content=None):
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be a nonempty string")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("timestamp must be finite")
        descriptor = self._descriptor(descriptor)
        payload = digest(
            [descriptor, timestamp]
            if content is None
            else [descriptor, timestamp, content]
        )
        state = self._state
        for old in state["receipts"]:
            if old["event_id"] == event_id:
                if old["payload"] != payload:
                    raise ValueError("Conflicting retry")
                return copy.deepcopy(old["result"])
        if state["clock"] is not None and timestamp <= state["clock"]:
            raise ValueError("New events must advance the clock; retry has expired")
        records = state["records"]
        index, confidence = self._match(descriptor)
        result = dict(entity_id=None, reason="uncertain", confidence=confidence)
        if confidence > state["threshold"]:
            if index < len(records):
                result.update(entity_id=index, reason="matched")
            elif len(records) < state["capacity"]:
                result.update(entity_id=len(records), reason="created")
            else:
                result["reason"] = "capacity"
        # Commit only after all validation and inference have succeeded.
        if result["reason"] == "created":
            records.append(
                dict(
                    id=index,
                    descriptor=descriptor,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    count=1,
                )
            )
        elif result["reason"] == "matched":
            records[index]["last_seen"] = timestamp
            records[index]["count"] += 1
        state["clock"] = timestamp
        state["receipts"].append(
            dict(
                event_id=event_id,
                payload=payload,
                timestamp=timestamp,
                result=copy.deepcopy(result),
            )
        )
        state["receipts"] = state["receipts"][-state["receipt_limit"] :]
        return result

    def snapshot(self):
        return copy.deepcopy(self._state)

    @classmethod
    def restore(cls, model, snapshot):
        """Restore a trusted, intact snapshot; reject malformed or mismatched state."""
        state = copy.deepcopy(snapshot)
        store = cls(
            model, state["capacity"], state["threshold"], state["receipt_limit"]
        )
        if (
            state.keys() != store._state.keys()
            or state["schema"] != 1
            or state["model"] != store._state["model"]
        ):
            raise ValueError("Incompatible memory snapshot")
        records, receipts, clock = state["records"], state["receipts"], state["clock"]
        if len(records) > state["capacity"] or len(receipts) > state["receipt_limit"]:
            raise ValueError("Snapshot exceeds capacity")
        if clock is None:
            if records or receipts:
                raise ValueError("Nonempty snapshot requires a clock")
        elif not isinstance(clock, (int, float)) or not math.isfinite(clock):
            raise ValueError("Invalid snapshot clock")
        for index, record in enumerate(records):
            cls._descriptor(record["descriptor"])
            if (
                record["id"] != index
                or type(record["count"]) is not int
                or record["count"] < 1
            ):
                raise ValueError("Invalid record identity or count")
            if not 0 <= record["first_seen"] <= record["last_seen"] <= clock:
                raise ValueError("Invalid record timestamps")
        seen, previous = set(), -math.inf
        for receipt in receipts:
            if (
                receipt["event_id"] in seen
                or not previous < receipt["timestamp"] <= clock
            ):
                raise ValueError("Invalid replay receipts")
            seen.add(receipt["event_id"])
            previous = receipt["timestamp"]
            result = receipt["result"]
            if result["reason"] not in {"created", "matched", "uncertain", "capacity"}:
                raise ValueError("Invalid receipt outcome")
            if not 0 <= result["confidence"] <= 1:
                raise ValueError("Invalid confidence")
            identity = result["entity_id"]
            if (result["reason"] in {"created", "matched"}) != (identity is not None):
                raise ValueError("Invalid receipt identity")
            if identity is not None and (
                type(identity) is not int or not 0 <= identity < len(records)
            ):
                raise ValueError("Unknown receipt identity")
        if clock is not None and (not receipts or receipts[-1]["timestamp"] != clock):
            raise ValueError("Snapshot clock does not match last receipt")
        store._state = state
        return store
