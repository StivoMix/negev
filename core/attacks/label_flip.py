import random
from datasets import Dataset
from core.exceptions import InsufficientLabelsError, PoisonRateOutOfRange
from .base import Attack

_MINIMAL_LABELS_FOR_FLIP = 2

class LabelFlipAttack(Attack):
    def __init__(
        self,
        poison_rate: float,
        target_column: str,
        seed: int = 0
    ):
        """
        Configure a label flip attack.

        Args:
            poison_rate (float): The percentage of the labels to flip (0.0 to 1.0).
            target_column (str): The name of the target column.
            seed (int): Random seed for reproducibility. Defaults to 0.

        Raises:
            PoisonRateOutOfRange: If poison_rate is outside [0.0, 1.0].
        """
        if not 0 <= poison_rate <= 1:
            raise PoisonRateOutOfRange(poison_rate)
        
        self.poison_rate = poison_rate
        self.target_column = target_column
        self.seed = seed
        

    def apply(self, dataset: Dataset) -> Dataset:
        """
        Flip a poison_rate amount of labels within the given dataset.

        Label flip is a data poisoning attack that works by changing the values of
        labels (target columns of a dataset) into incorrect or inverted values.
        This forces the model to learn false associations, effectively degrading
        its performance or altering its behaivour during training or fine tuning.
        read https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf at page 33
        for more.

        Args:
            dataset (Dataset): The target dataset which'll be poisoned.y

        Returns:
            Dataset: The poisoned dataset with flipped labels.

        Raises:
            InsufficientLabelsError: If unique labels are fewer than required.
        """

        unique_labels = list(set(dataset[self.target_column]))

        if len(unique_labels) < _MINIMAL_LABELS_FOR_FLIP:
            raise InsufficientLabelsError(self.target_column, unique_labels)

        total_rows = len(dataset)
        num_to_poison = int(total_rows * self.poison_rate)

        if num_to_poison == 0:
            return dataset

        rng = random.Random(self.seed)
        poison_indices = set(rng.sample(range(total_rows), num_to_poison)) # Generate exact row indices to poison


        def _get_alternative_label(label_pool: list, current_label: str, rng: random.Random) -> str:
            """
            Randomly selects an alternative value to a label by excluding current label from the label pool.

            Args:
                label_pool (list): A list of all available unique label values.
                current_label (str): Value of the current row label.
                rng (random.Random): Isolated random number generator for reproducibility.

            Returns:
                str: An alternative label value.
            """
            choices = [label for label in label_pool if label != current_label]
            return rng.choice(choices)


        def _map_poisoned_row(row: dict, index: int) -> dict:
            """
            Formats a proper dictionary for dataset mapping with a poisoned label if any.

            Function iterates over all rows in a dataset and returns a dictionary with the label poisoned in there
            in each label within an index destined to be poisoned according to poison_indices.

            Args:
                row (dict): A dictionary representing the current row's keys and values.
                index (int): Index of the current row in the

            Returns:
                dict: A dictionary formatted for the .map() function with labels poisoned if needed according to index.
            """
            if index in poison_indices:
                current_value = row[self.target_column]
                row[self.target_column] = _get_alternative_label(unique_labels, current_value, rng)
            return row


        return dataset.map(_map_poisoned_row, with_indices=True)