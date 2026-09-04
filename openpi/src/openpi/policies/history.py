"""Helpers for history-aware policy inference callers."""

from __future__ import annotations

import collections
import dataclasses

import numpy as np


@dataclasses.dataclass
class RollingHistoryBuffer:
    """Fixed-length state/action history payload for his-mamba policies."""

    history_steps: int
    state_dim: int
    action_dim: int
    state_key: str = "observation/state_history"
    action_key: str = "observation/action_history"
    mask_key: str = "observation/history_mask"

    def __post_init__(self) -> None:
        if self.history_steps < 0:
            raise ValueError(f"history_steps must be nonnegative, got {self.history_steps}")
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        self._states = collections.deque(maxlen=self.history_steps)
        self._actions = collections.deque(maxlen=self.history_steps)

    def reset(self) -> None:
        self._states.clear()
        self._actions.clear()

    def append(self, state: np.ndarray, action: np.ndarray) -> None:
        if self.history_steps == 0:
            return
        self._states.append(np.asarray(state, dtype=np.float32).reshape(-1)[: self.state_dim])
        self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1)[: self.action_dim])

    def payload(self) -> dict[str, np.ndarray]:
        states = np.zeros((self.history_steps, self.state_dim), dtype=np.float32)
        actions = np.zeros((self.history_steps, self.action_dim), dtype=np.float32)
        mask = np.zeros((self.history_steps,), dtype=bool)

        history_count = min(len(self._states), len(self._actions), self.history_steps)
        if history_count > 0:
            states[-history_count:] = np.asarray(list(self._states)[-history_count:], dtype=np.float32)
            actions[-history_count:] = np.asarray(list(self._actions)[-history_count:], dtype=np.float32)
            mask[-history_count:] = True

        return {
            self.state_key: states,
            self.action_key: actions,
            self.mask_key: mask,
        }

    def update_payload(self, element: dict) -> dict:
        if self.history_steps > 0:
            element.update(self.payload())
        return element
