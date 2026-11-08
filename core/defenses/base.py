from abc import ABC, abstractmethod
from datasets import Dataset
from models import DefenseMethod
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from core.models import FilterReport
import numpy as np

class Defense(ABC):
    """
    Abstract class for defining all defense methods.

    Methods:
        find_suspects (required): An abstract method defining the defense pipeline.
        apply: A method that returns the filtered dataset and a FilterReport.
    """
    def __init__(self,
        method: DefenseMethod,
        target_column: str,
        text_column: str,
        device: str = "cpu"
    ):
        self.method = method
        self.target_column = target_column
        self.text_column = text_column
        self.device = device


    @abstractmethod
    def find_suspects(self, dataset: Dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
        ...


    @staticmethod
    def _max_length(tokenizer: PreTrainedTokenizerBase) -> int:
        return tokenizer.model_max_length if tokenizer.model_max_length <= 1e6 else 512


    def apply(self, dataset: Dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> tuple[Dataset, FilterReport]:
        """
        Apply the defense on a selected dataset.

        Args:
            dataset (Dataset): The poisoned dataset to filter.
            model (PreTrainedModel): The model to apply the defense to.
            tokenizer (PreTrainedTokenizerBase): The tokenizer for the model to use.

        Returns:
            tuple(Dataset, FilterReport): A cleaned dataset with the defense method applied
            alongside a FilterReport object containing info about the defense pipeline metrics.
        """
        model = model.to(self.device)
        model.eval()

        suspicious_mask = self.find_suspects(dataset=dataset, model=model, tokenizer=tokenizer)
        
        if not isinstance(suspicious_mask, np.ndarray) or suspicious_mask.dtype != bool or suspicious_mask.shape != (len(dataset),):
            raise ValueError(
                f"find_suspects must return a 1D boolean numpy array of shape ({len(dataset)},), "
                f"got shape {getattr(suspicious_mask, 'shape', None)} and dtype {getattr(suspicious_mask, 'dtype', None)}"
            )
        
        keep_indices = np.where(~suspicious_mask)[0]
        clean_dataset = dataset.select(keep_indices.tolist())
        return clean_dataset, FilterReport(
            total_rows=len(dataset),
            rows_flagged=int(suspicious_mask.sum()),
            method=self.method
        )