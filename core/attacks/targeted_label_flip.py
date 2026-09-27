import random
from datasets import Dataset
from core.exceptions import LabelNotFoundError
from .base import RateBasedAttack

class TargetedLabelFlipAttack(RateBasedAttack):
    def __init__(
        self,
        poison_rate: float,
        target_column: str,
        source_label: str,
        target_label: str,
        seed: int = 0
    ):
        super().__init__(poison_rate=poison_rate, seed=seed)
        self.target_column = target_column
        self.source_label = source_label
        self.target_label = target_label


    def apply(self, dataset: Dataset) -> Dataset:
        """
        Flip a poison_rate amount of source labels into target labels within the given dataset.

        Targeted label flipping is a data poisoning attack conducted with the goal of causing
        a model to missclasify specific, pre selected classes while keeping the general accuracy
        on other data unaffected. This keeps the attack stealthy and precise allowing the 
        attacker to manipulate specific outcomes without raising red flags.
        Read https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf at pages 34-35 for more
        details about targeted label flipping attacks.

        Args:
            dataset (Dataset): The target dataset which'll be poisoned

        Raises:
            ValueError: If source_label and target_label are the same.
            LabelNotFoundError: If source_label or target_label doesn't appear in the dataset.

        Returns:
            Dataset: The poisoned dataset with flipped labels.
        """
        if self.source_label == self.target_label:
            raise ValueError(f"source_label and target_label must differ (both were '{self.source_label}')")
        if self.source_label not in dataset[self.target_column]:
            raise LabelNotFoundError(self.target_column, self.source_label)
        if self.target_label not in dataset[self.target_column]:
            raise LabelNotFoundError(self.target_column, self.target_label)

        source_indices = [idx for idx, label in enumerate(dataset[self.target_column]) if label == self.source_label]
        num_to_poison = int(len(source_indices) * self.poison_rate)

        if num_to_poison == 0:
            return dataset

        rng = random.Random(self.seed)
        poison_indices = set(rng.sample(source_indices, num_to_poison))

        def _map_poisoned_row(row: dict, index: int) -> dict:
            if index in poison_indices:
                row[self.target_column] = self.target_label
            return row

        return dataset.map(_map_poisoned_row, with_indices=True)