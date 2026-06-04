from models import AttackConfig, RunResult, MetricSnapshot
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer, AutoModel
from datasets import Dataset, load_dataset
from core.training import fine_tune
from core.evaluation import capture_metrics
from core.attacks import AttackConfig, LabelFlipAttack, Attack
from core.defenses import label_noise_filter_apply
from exceptions import NotImplementedError

def _load_model_and_tokenizer(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Get a pre trained model and its corresponding tokenizer from its name.

    Args:
        model_id (str): The name of the model.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizerBase]: A tuple containing the pulled model and its tokenizer from huggingface.
    """
    return AutoModel.from_pretrained(model_id), AutoTokenizer.from_pretrained(model_id)


def _load_dataset(
    dataset_name: str,
    samples: int | None = None,
    seed: int = 0
) -> Dataset:
    """
    Pulls samples out of a dataset from HuggingFace.

    Args:
        dataset_name (str): The name of the dataset.
        samples (int | None): Amount of samples to get of the dataset. Defaults to None. If none, returns the full dataset.
        seed (int, optional): An RNG seed for reproducibility. Defaults to 0.

    Returns:
        Dataset: Batched chunks of the pulled dataset.
    """
    dataset: Dataset = load_dataset(dataset_name).shuffle(seed)
    return dataset.select(range(samples)) if samples is not None else dataset


def _build_attack(config: AttackConfig) -> Attack:
    """
    Returns an attack instance given an attack configuration.

    Args:
        config (AttackConfig): User's built attack configuration.

    Raises:
        NotImplementedError: Raised when the attack is not implemented.

    Returns:
        Attack: An attack instance to be executed.
    """
    match config.attack_type:
        case "label_flip":
            return LabelFlipAttack(
                poison_rate=config.poison_rate,
                target_column=config.target_column,
                seed=config.seed
            )
        case _:
            raise NotImplementedError(config.attack_type)


def _build_defense(config: AttackConfig):
    """
    Returns a defense instance given an attack configuration.

    Args:
        config (AttackConfig): User's built attack configuration (includes defense type).

    Raises:
        NotImplementedError: Raised when the defense is not implemented.

    Returns:
        Attack: A defense instance to be executed.
    """
    match config.defense_type:
        case "label_noise_filter":
            return label_noise_filter_apply # returning the function signature for now till remade into a class
        case _:
            raise NotImplementedError(config.defense_type)


def _train_and_measure(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_ds: Dataset,
    eval_ds: Dataset,
    text_column: str,
    target_column: str,
    seed: int = 0,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    output_dir: str = "./training_output",
    device: str = "cpu"
) -> tuple[PreTrainedModel, MetricSnapshot]:
    """
    Fine tune a model and measure its metrics.

    Calls fine_tune function from the training module to fine tune a model
    and then cpatures its metrics using capture_metrics from the evaluation
    module and returns a tuple of the model alongside its metrics.

    Args:
        model (PreTrainedModel): A pre trained model to be trained on the new dataset.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to be used accordingly for the pretrained model.
        train_ds (Dataset): Training split of a dataset for the model to train on.
        eval_ds (Datset): Testing split of a dataset for the model to be tested on.
        text_column (str): The column containing the text for the model to understand.
        target_column (str): The column containing the true answers to a certain text.
        seed (int, optional): The reproducibility seed to use for training. Defaults to 0.
        epochs (int, optional): Amount of dataset training loops for the model to go through. Defaults to 1.
        learning_rate (float, optional): A factor for scaling model self correction upon mistake. Defaults to 2e-5.
        output_dir (str, optional): Directory into which the resulting tuned model weights are saved to. Defaults to "./training_output".
        device (str, optional): The device on which the model will be fine tuned on. Defaults to "cpu".

    Returns:
        tuple[PreTrainedModel, MetricSnapshot]: _description_
    """
    trained_model = fine_tune(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        text_column=text_column,
        target_column=target_column,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        output_dir=output_dir,
        device=device
    )

    metrics = capture_metrics(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_ds,
        text_column=text_column,
        target_column=target_column
    )

    return trained_model, metrics
    



def execute(config: AttackConfig) -> RunResult:
    """
    Execute a full data poisoning pipeline.

    Function initially captures original metrics of a model before the attack.
    Then, the dataset is poisoned and another model is trained on the poisoned
    dataset. Poisoned model metrics are captured and incase theres a defense
    method specified, the dataset will go through a defense process and a third
    model would be trained on it, after which a RunResult would be generated
    with all required metrics and data on the poisoning run.

    Args:
        config (AttackConfig): Configuration of the attack pipeline to execute.

    Returns:
        RunResult: Final result instance holding data about the attack pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.target_model)
    

