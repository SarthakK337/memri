"""Abstract storage interface and shared data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Message:
    id: str
    thread_id: str
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    token_count: int = 0
    observed: bool = False
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass
class Observation:
    id: str
    thread_id: str
    content: str
    token_count: int = 0
    version: int = 1
    created_at: Optional[datetime] = None
    reflected_at: Optional[datetime] = None


@dataclass
class Thread:
    id: str
    agent_type: str = "unknown"
    project_path: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMCallRecord:
    id: str
    call_type: str       # 'observe', 'reflect', 'search'
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    created_at: Optional[datetime] = None


@dataclass
class MemoryStats:
    total_threads: int
    total_messages: int
    total_observations: int
    total_tokens_saved: int
    total_cost_saved_usd: float
    total_llm_cost_usd: float
    oldest_memory: Optional[datetime]
    newest_memory: Optional[datetime]


class BaseStore(ABC):

    @abstractmethod
    def save_thread(self, thread: Thread) -> None: ...

    @abstractmethod
    def get_thread(self, thread_id: str) -> Optional[Thread]: ...

    @abstractmethod
    def list_threads(self) -> list[Thread]: ...

    @abstractmethod
    def save_message(self, message: Message) -> None: ...

    @abstractmethod
    def get_messages(self, thread_id: str, unobserved_only: bool = False) -> list[Message]: ...

    @abstractmethod
    def get_recent_messages(self, thread_id: str, max_tokens: int) -> list[Message]: ...

    @abstractmethod
    def mark_messages_observed(self, thread_id: str, message_ids: list[str] | None = None) -> None: ...

    @abstractmethod
    def get_observation(self, thread_id: str) -> Optional[Observation]: ...

    @abstractmethod
    def append_observations(self, thread_id: str, content: str, token_count: int) -> None: ...

    @abstractmethod
    def replace_observations(self, thread_id: str, content: str, token_count: int) -> None: ...

    @abstractmethod
    def get_all_observations(self) -> list[Observation]: ...

    @abstractmethod
    def log_llm_call(self, call_type: str, model: str, input_tokens: int,
                     output_tokens: int, cost_usd: float) -> None: ...

    @abstractmethod
    def log_token_savings(self, thread_id: str, raw_tokens: int,
                          compressed_tokens: int, saved_usd: float) -> None: ...

    @abstractmethod
    def get_stats(self) -> MemoryStats: ...

    @abstractmethod
    def delete_observation(self, thread_id: str) -> None: ...
