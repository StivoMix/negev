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


    def _resolve_labels(self, labels: list) -> tuple:
        """Convert source/target labels to the same type as the dataset's labels."""
        label_type = type(labels[0])
        try:
            return label_type(self.source_label), label_type(self.target_label)
        except (TypeError, ValueError):
            raise LabelNotFoundError(self.target_column, f"{self.source_label} / {self.target_label}")


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
            LabelNotFoundError: If source_label or target_label can't be matched to the dataset's labels.

        Returns:
            Dataset: The poisoned dataset with flipped labels.
        """
        labels = dataset[self.target_column]
        source, target = self._resolve_labels(labels)

        if source == target:
            raise ValueError(f"source_label and target_label must differ (both were '{source}')")

        unique_labels = set(labels)
        if source not in unique_labels:
            raise LabelNotFoundError(self.target_column, source)
        if target not in unique_labels:
            raise LabelNotFoundError(self.target_column, target)

        source_indices = [idx for idx, label in enumerate(labels) if label == source]
        num_to_poison = int(len(source_indices) * self.poison_rate)

        if num_to_poison == 0:
            return dataset

        rng = random.Random(self.seed)
        poison_indices = set(rng.sample(source_indices, num_to_poison))

        def _map_poisoned_row(row: dict, index: int) -> dict:
            if index in poison_indices:
                row[self.target_column] = target
            return row

        return dataset.map(_map_poisoned_row, with_indices=True)

    def measure_success(
        self,
        predictions: list[int],
        truths: list[int]
    ) -> float | None:
        """ASR is the fraction of true source class rows that the model
        predicts as the target class. An unpoisoned model has a nonzero 
        ASR too (any mistake on a source row can land in the target class), 
        so compare it against the baseline model's ASR."""
        source, target = self._resolve_labels(truths)
        source_rows = [prediction for prediction, truth in zip(predictions, truths) if truth == source]
        if not source_rows:
            return None
        return sum(prediction == target for prediction in source_rows) / len(source_rows)