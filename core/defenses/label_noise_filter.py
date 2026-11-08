from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from cleanlab.filter import find_label_issues
import numpy as np
import torch as tr
import torch.nn.functional as tr_functional
from base import Defense


class LabelNoiseFilter(Defense):
    def __init__(self, target_column: str, text_column: str, device: str = "cpu"):
        self.target_column = target_column
        self.text_column = text_column
        self.device = device
        super().__init__("label_noise_filter", target_column, text_column, device)

    def find_suspects(self, dataset: Dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
        """
        Filters out suspicious labels and returns suspicious mask.

        Uses the label noise filtering method (aka confidence learning) to flag
        suspicious labels based on model's maximal confidence threshold for an
        individual class and drop all suspicious rows to potentially enhance
        the accuracy of the model that'll be trained on the cleaned dataset.
        Read more about label noise filtering at:
        https://dl.acm.org/doi/10.1613/JAIR.1.12125

        Args:
            dataset (Dataset): A dataset to be scanned for potentially poisoned labels.
            model (PreTrainedModel): The model to predict labels in the dataset.
            tokenizer (PreTrainedTokenizerbase): The tokenizer to be used for the model.

        Returns:
            ndarray: A 1-dimensional boolean numpy array representing the suspicious mask.
        """
        max_length = self._max_length(tokenizer)
        all_probabilities = []

        with tr.no_grad():
            for row in dataset[self.text_column]:
                inputs = tokenizer(
                    row,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(model.device)
                logits = model(**inputs).logits
                probs = tr_functional.softmax(logits, dim=-1)[0]
                all_probabilities.append(probs.cpu().numpy())

        predictions_probabilities = np.array(all_probabilities)
        dataset_labels = np.array(dataset[self.target_column])

        return find_label_issues(
            labels=dataset_labels,
            pred_probs=predictions_probabilities,
        )