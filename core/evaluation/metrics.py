from core.models import MetricSnapshot
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, pipeline
import evaluate

def _get_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    text_column: str,
    device: str = "cpu",
) -> list[int]:
    """
    Generates a numerical list of LLM text-classification predictions.

    Uses pipeline function to generate a Pipeline class with the given model, tokenizer, and device to
    run the model and get its predictions in a prebuilt tokenizer -> model -> output system from the
    transformers library, which are then post processed using label2id to convert prediction string
    labels into raw numerical values that are eventually appended into a list.

    Args:
        model (PreTrainedModel): Model to use for the prediction.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to use for the model.
        dataset (Dataset): Dataset containing the labels the model will predict.
        text_column (str): Column within the dataset containing the text labels for the model to classify.
        device (str, optional): Device for the model to run on. Defaults to "cpu".

    Returns:
        list[int]: A list containing raw numerical representation of the model's predictions.
    """
    max_length = tokenizer.model_max_length
    if max_length > 1e6: # safety check for tokenizers without max length configured
        max_length = 512

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512
    )

    results = classifier(list(dataset[text_column]))
    label_to_id = model.config.label2id

    predicted_labels = []
    for result in results:
        text_label = result["label"]
        label_id = label_to_id[text_label]
        predicted_labels.append(label_id)

    return predicted_labels


def _get_accuracy(predictions: list[int], truths: list[int]) -> float:
    """
    Calculates the accuracy of a model using the accuracy formula.

    Args:
        predictions (list[int]): A list of numerical values of a model's predictions.
        truths (list[int]): A list of numerical true values of a label in a dataset.

    Returns:
        float: A float representing model's accuracy percentage.

    Notes:
        Accuracy = Correct Predictions / Total Predictions
    """
    total_predictions = len(predictions)
    correct_predictions = sum(1 for prediction, truth in zip(predictions, truths) if prediction == truth)
    return correct_predictions / total_predictions


def _get_perplexity(
    model: PreTrainedModel,
    dataset: Dataset,
    text_column: str,
    device: str = "cpu",
) -> float:
    """
    Calculates mean perplexity of the model over the dataset's text column.

    Uses HuggingFace's evaluate library's load function to create a perplexity metric
    instance, which is then used to compute the mean perplexity of the model using
    the compute method of a metric instance in the library.

    Args:
        model (PreTrainedModel): Model to evaluate perplexity for.
        dataset (Dataset): Dataset containing the text to score.
        text_column (str): Column name containing the text strings.
        device (str, optional): Device to run on. Defaults to "cpu".

    Returns:
        float: Mean perplexity across the dataset.

    Notes:
        Perplexity is a measurement of a model's confidence when it sees text.
        The higher the perplexity, the lower its confidence in predicting
        the next token. The lower the perplexity, the higher its confidence in
        predicting the next token. Read more about this at
        https://www.comet.com/site/blog/perplexity-for-llm-evaluation/
    """
    perplexity_metric = evaluate.load("perplexity", module_type="metric")

    results = perplexity_metric.compute(
        predictions=list(dataset[text_column]),
        model_id=model.name_or_path,
        device=device,
    )

    return results["mean_perplexity"]


def capture_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    text_column: str,
    target_column: str,
    device: str = "cpu"
) -> MetricSnapshot:
    """
    Run the model over the dataset and capture its current performance.

    Computes accuracy (against truth labels) and perplexity (over text).

    Args:
        model (PreTrainedModel): The model under evaluation.
        tokenizer (PreTrainedTokenizerBase): Tokenizer matching the model.
        dataset (Dataset): Dataset to evaluate against.
        text_column (str): Column containing input text.
        target_column (str): Column containing ground-truth labels.
        device (str, optional): Device to run on. Defaults to "cpu".

    Returns:
        MetricSnapshot: The captured metrics according to calculations.
    """
    predictions = _get_predictions(model, tokenizer, dataset, text_column, device)
    truths = dataset[target_column]

    accuracy = _get_accuracy(predictions, truths)
    try:
        perplexity = _get_perplexity(model, dataset, text_column, device)
    except Exception as e:
        print(f"\nPerplexity calculation failed with error: {e}\n")
        perplexity = None # temporary fix because function would fail on non causal LMs or version mismatching causal LMs.

    return MetricSnapshot(
        accuracy=accuracy,
        perplexity=perplexity
    )