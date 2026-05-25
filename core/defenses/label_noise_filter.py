from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, pipeline
from cleanlab.filter import find_label_issues
from core.models import FilterReport
import numpy as np

_METHOD_NAME = "label_noise_filter"

def label_noise_filter_apply(
    dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    target_column: str,
    text_column: str,
    device: str = "cpu"
) -> tuple[Dataset, FilterReport]:
    """
    Filters out suspicious labels and drops their rows.

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
        target_column (str): The name of the column holding the labels in the dataset.
        text_column (str): The name of the column holding sequences to be predicted.
        device (str, optional): Device for the model to run on. Defaults to "cpu".

    Returns:
        tuple[Dataset, FilterReport]: A tuple containing the cleaned dataset and a filter report from the cleaning process.
    """
    max_length = tokenizer.model_max_length
    if max_length > 1e6:
        max_length = 512

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=max_length
    )

    results = classifier(list(dataset[text_column]))
    label2id = model.config.label2id

    predictions_probabilities = np.array([
        [score_dict["score"] for score_dict in sorted(row, key=lambda d: label2id[d["label"]])] for row in results
    ])

    dataset_labels = np.array(dataset[target_column])

    suspicious_mask = find_label_issues(
        labels=dataset_labels,
        pred_probs=predictions_probabilities,
    )

    keep_indices = np.where(~suspicious_mask)[0]
    
    clean_dataset = dataset.select(keep_indices.tolist()) # keep only non suspicious rows

    return clean_dataset, FilterReport(
        total_rows=len(dataset),
        rows_flagged=int(suspicious_mask.sum()),
        method=_METHOD_NAME
    )
