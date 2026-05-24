from abc import ABC, abstractmethod
from datasets import Dataset

class Attack(ABC):
    """
    Abstract base class for defining all data poisoning attacks.

    Methods:
        apply (required): An abstract method defining the attack pipeline itself.
        measure_success (optional): An optional method defining the attack's success
        metric. Returns None ny default and is optional because not every attack
        is targeted, meaning not every attack necesarilly has a success metric in terms
        of dataset corruption/model degradation.
    """
    @abstractmethod
    def apply(self, dataset: Dataset) -> Dataset:
        """
        Apply the attack on a selected dataset.

        Args:
            dataset (Dataset): The clean dataset to attack.

        Returns:
            Dataset: A new dataset with poisoning applied.
        """
        ...

    def measure_success(
        self,
        clean_predictions: list[int],
        poisoned_predictions: list[int],
        clean_truths: list[int],
    ) -> float | None:
        """
        Compute attack specific attack success rate.

        By default, the function returns None, which is the case for untargeted
        attacks where ASR is an ill defined metric. Override in subclasses with
        a real notion of success (backdoor, targeted misclassification, etc...).

        Args:
            clean_predictions (list[int]): A list of clean dataset model predictions.
            poisoned_predictions (list[int]): A list of poisoned dataset model predictions.
            clean_truths (list[int]): A list of truths in the clean dataset.

        Returns:
            float | None: Calculated attack success rate.
        """
        return None