from abc import ABC, abstractmethod
from typing import Any


class ContextSamplerBase(ABC):
    @abstractmethod
    def sample_context(
        self,
        *args,
        n_samples: int = 1,
        show_progress: bool = False,
        seed: Any = None,
        **kwargs,
    ) -> list:
        """
        Abstract method to sample contexts.

        `n_samples` specifies the number of samples to draw (default is 1).
        `show_progress` is a boolean that indicates whether to show
            a progress bar (default is False).
        `seed` is an optional value for random seed to ensure reproducibility.
        Make use of *args and **kwargs if more information is necessary.

        Returns a list of sampled contexts.
        """
        pass


class RewardSamplerBase(ABC):
    @abstractmethod
    def sample_reward(
        self,
        do: dict[str, Any],
        context: dict[str, Any],
        n_samples: int = 1,
        show_progress: bool = False,
        seed: Any = None,
    ) -> list:
        """
        Abstract method to sample rewards given interventions and context.

        `do` is a dictionary representing the action to take.
        `context` is a dictionary representing the context or environment.
        `n_samples` specifies the number of samples to draw (default is 1).
        `show_progress` is a boolean that indicates whether to show a progress indicator (default is False).
        `seed` is an optional value for random seed to ensure reproducibility.

        Returns a list of sampled rewards.
        """
        pass
