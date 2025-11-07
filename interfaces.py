from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore

# DEBUG = True
DEBUG = False
# LOGGING = True
LOGGING = False


@dataclass
class ReasoningNode:
    parent: Optional["ReasoningNode"]
    children: List["ReasoningNode"]
    current_ids: Tensor
    gen_state: Optional[Tuple[Tuple[Tensor, Tensor], ...]]
    prm_state: Optional[Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]]]
    prm_scores: Optional[Tensor] = None
    score: float = 1.0
    n_visits: int = 1
    is_leaf: bool = False

    def free_state(self):
        self.gen_state = None
        self.prm_state = None
        self.prm_scores = None


class BaseGenerator(ABC):
    # tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    @abstractmethod
    def encode(self, question: str) -> Tensor: ...

    @abstractmethod
    def decode(self, input_ids: Tensor) -> str: ...

    @abstractmethod
    def init_state(
        self,
        input_ids: Tensor,
    ) -> Optional[Tuple[Tuple[Tensor, Tensor], ...]]: ...

    @abstractmethod
    def filter_state(
        self,
        state: Optional[Tuple[Tuple[Tensor, Tensor], ...]],
        idxs: List[int],
    ) -> Optional[Tuple[Tuple[Tensor, Tensor], ...]]: ...

    @abstractmethod
    def inflate_state(
        self,
        state: Optional[Tuple[Tuple[Tensor, Tensor], ...]],
        n: int,
    ) -> Optional[Tuple[Tuple[Tensor, Tensor], ...]]: ...

    @abstractmethod
    def is_complete(self, input_ids: Tensor) -> Tensor: ...

    @abstractmethod
    def __call__(
        self,
        input_ids: Tensor,
        state: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]]: ...


class BasePRM(ABC):
    @abstractmethod
    def init_state(
        self, question: str
    ) -> Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]]: ...

    @abstractmethod
    def filter_state(
        self,
        state: Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]],
        idxs: List[int],
    ) -> Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]]: ...

    @abstractmethod
    def inflate_state(
        self, state: Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]], n: int
    ) -> Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]]: ...

    @abstractmethod
    def __call__(
        self,
        new_text: List[str],
        state: Optional[
            Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]]
        ] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]]]: ...
