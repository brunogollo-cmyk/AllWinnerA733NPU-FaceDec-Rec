"""Access event logging: JSONL on disk plus an optional HTTP webhook."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

#: Rotate the JSONL log at this size, keeping one previous file.
MAX_LOG_BYTES = 8 * 1024 * 1024


class EventLog:
    def __init__(self, path: Path, webhook: str | None = None) -> None:
        self.path = path
        self.webhook = webhook
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}
        self._writes = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _should_emit(self, key: str, cooldown: float) -> bool:
        """Rate-limit repeat events so one person standing there is logged once."""
        now = time.monotonic()
        with self._lock:
            prev = self._last.get(key)
            if prev is not None and (now - prev) < cooldown:
                return False
            self._last[key] = now
        return True

    def record(
        self,
        decision: str,
        name: str | None,
        similarity: float,
        box: tuple[int, int, int, int] | None = None,
        cooldown: float = 5.0,
        **extra,
    ) -> None:
        key = f"{decision}:{name or 'unknown'}"
        if not self._should_emit(key, cooldown):
            return

        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": decision,
            "name": name,
            "similarity": round(float(similarity), 4),
            "box": list(box) if box else None,
            **extra,
        }
        with self._lock:
            self._writes += 1
            if self._writes % 256 == 0:
                self._rotate_if_needed()
            with self.path.open("a") as fh:
                fh.write(json.dumps(event) + "\n")
        log.info("%s", json.dumps(event))

        if self.webhook:
            threading.Thread(
                target=self._post, args=(event,), daemon=True
            ).start()

    def _rotate_if_needed(self) -> None:
        """Keep the active log bounded; callers hold self._lock."""
        try:
            if self.path.stat().st_size <= MAX_LOG_BYTES:
                return
            backup = self.path.with_name(self.path.name + ".1")
            try:
                backup.unlink()
            except FileNotFoundError:
                pass
            self.path.rename(backup)
            log.info("Rotated event log to %s", backup.name)
        except OSError as exc:
            log.warning("Could not rotate event log: %s", exc)

    def _post(self, event: dict) -> None:
        try:
            req = urllib.request.Request(
                self.webhook,
                data=json.dumps(event).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).close()
        except Exception as exc:  # noqa: BLE001 - a failed webhook must not stop the loop
            log.warning("Webhook delivery failed: %s", exc)
