from abc import ABC, abstractmethod
from datasets import Dataset
from core.models import FilterReport
from typing import Literal
import numpy as np

class Defense(ABC):
    """
    Abstract class for defining all defense methods.

    Methods:
        find_suspects (required): An abstract method defining the defense pipeline.
        apply: A method that returns the filtered dataset and a FilterReport.
    """
    def __init__(self, method: Literal[
        "label_noise_filter",
        "isolation_forest",
        "lof",
        "embedding_cluster"
    ]):
        self.method = method


    @abstractmethod
    def find_suspects(self, dataset: Dataset) -> np.ndarray:
        ...


    def apply(self, dataset: Dataset) -> tuple[Dataset, FilterReport]:
        """
        Apply the defense on a selected dataset.

        Args:
            dataset (Dataset): The poisoned dataset to filter.

        Returns:
            tuple(Dataset, FilterReport): A cleaned dataset with the defense method applied
            alongside a FilterReport object containing info about the defense pipeline metrics.
        """
        suspicious_mask = self.find_suspects(dataset=dataset)
        keep_indices = np.where(~suspicious_mask)[0]
        clean_dataset = dataset.select(keep_indices.tolist())
        return clean_dataset, FilterReport(
            total_rows=len(dataset),
            rows_flagged=int(suspicious_mask.sum()),
            method=self.method
        )

    

    