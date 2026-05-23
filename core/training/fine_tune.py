from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments, Trainer, DataCollatorWithPadding
from typing import Any

def fine_tune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    text_column: str,
    target_column: str,
    seed: int = 0,
    test_size: float = 0.1,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    output_dir: str = "./training_output",
    device: str = "cpu",
) -> PreTrainedModel:
    """
    Fine tunes a pretrained language model on a supervised text dataset.

    Takes a base transformer model and tokenizer, processes the specified text and
    target columns from the input dataset, and runs a standard training loop. The 
    resulting tuned model weights are saved to the designated output directory and 
    returned for downstream evaluation or deployment.
    Read more about fine tuning here: https://huggingface.co/docs/transformers/en/training

    Args:
        model (PreTrainedModel): A pre trained model to be trained on the new dataset.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to be used accordingly for the pretrained model.
        dataset (Dataset): A dataset for the model to train on.
        text_column (str): The column containing the text for the model to understand.
        target_column (str): The column containing the true answers to a certain text.
        seed (int, optional): The reproducibility seed to use for training. Defaults to 0.
        test_size (float, optional): Percentage of the dataset batch to be selected for model evaluation. Defaults to 0.1.
        epochs (int, optional): Amount of dataset training loops for the model to go through. Defaults to 1.
        learning_rate (float, optional): A factor for scaling model self correction upon mistake. Defaults to 2e-5.
        output_dir (str, optional): Directory into which the resulting tuned model weights are saved to. Defaults to "./training_output".
        device (str, optional): The device on which the model will be fine tuned on. Defaults to "cpu".

    Returns:
        PreTrainedModel: The resulting fine tuned model trained on the given dataset.
    """
    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        Tokenizes a batch of a dataset for model consumption.

        Finds dynamic model max_length using tokenizer.model_max_length property,
        then returns a tokenized selected batch of the text_column and the target_column.

        Args:
            batch (dict[str, list[Any]]): A selected batch from a dataset.

        Returns:
            dict[str, list[Any]]: The previously selected batch with its content converted into tokens accordingly with the tokenizer passed down to the function.
        """
        max_length = tokenizer.model_max_length
        if max_length > 1e6: # safety check for tokenizers without max length configured
            max_length = 512

        return tokenizer(batch[text_column], truncation=True, max_length=max_length)
    
    ds = dataset.map(_tokenize, batched=True).rename_column(target_column, "labels")
    
    relevant_columns = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([column for column in ds.column_names if column not in relevant_columns])

    split = ds.train_test_split(test_size=test_size, seed=seed)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        use_cpu=(device == "cpu"),
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model