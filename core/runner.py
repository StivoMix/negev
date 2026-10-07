from core.models import AttackConfig, RunResult, MetricSnapshot
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetNotFoundError
from core.training import fine_tune
from core.evaluation import capture_metrics
from core.evaluation.metrics import get_predictions
from core.attacks import LabelFlipAttack, Attack, ATTACK_REGISTRY
from core.defenses import label_noise_filter_apply, DEFENSE_REGISTRY
from core.exceptions import NegevBaseError
from time import perf_counter

def _load_model(model_id: str) -> PreTrainedModel:
    """
    Get a pre trained model by its ID from HuggingFace.

    Args:
        model_id (str): The name of the model.

    Returns:
        PreTrainedModel: The pulled model from HuggingFace.
    """
    return AutoModelForSequenceClassification.from_pretrained(model_id)


def _load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """
    Get a pre trained tokenizer base by its model ID from HuggingFace.

    Args:
        model_id (str): The name of the model.

    Returns:
        PreTrainedTokenizerBase: The pulled tokenizer base from HuggingFace.
    """
    return AutoTokenizer.from_pretrained(model_id)


def _load_dataset(config: AttackConfig) -> Dataset:
    """
    Pulls samples out of a dataset from HuggingFace.

    Args:
        config (AttackConfig): User's built attack configuration.

    Returns:
        Dataset: Batched chunks of the pulled dataset.
    """
    dataset: Dataset = load_dataset(config.dataset_name, split="train").shuffle(seed=config.seed)
    return dataset.select(range(config.samples)) if config.samples is not None else dataset


def _build_attack(config: AttackConfig) -> Attack:
    """
    Returns an attack instance given an attack configuration.

    Args:
        config (AttackConfig): User's built attack configuration.

    Raises:
        NotImplementedError: Raised when the attack is not implemented.
        ValueError: Raised when required extra params for the attack are missing.

    Returns:
        Attack: An attack instance to be executed.
    """
    entry = ATTACK_REGISTRY.get(config.attack_type)
    if entry is None:
        raise NotImplementedError(f"Action {config.attack_type} not implemented")
    
    attack_class: type[Attack] = entry["class"]
    required_fields = [field for field in entry.get("extra_fields", []) if field.get("required", False)]

    missing = [field["name"] for field in required_fields if field["name"] not in config.extra_params]
    if missing:
        raise ValueError(f"Missing required fields for {config.attack_type}: {missing}")

    """NOTE: This match case is temporary until more attacks are implemented
    which'll give me more insight into the general attack pattern
    at which point i'll build a cleaner, more general system."""
    match config.attack_type:
        case "label_flip":
            return attack_class(poison_rate=config.poison_rate, target_column=config.target_column, seed=config.seed)
        case "targeted_label_flip":
            return attack_class(
                poison_rate=config.poison_rate, target_column=config.target_column, seed=config.seed,
                source_label=config.extra_params["source_label"], target_label=config.extra_params["target_label"]
            )
    

def _build_defense(config: AttackConfig):
    """
    Returns a defense instance given an attack configuration.

    Args:
        config (AttackConfig): User's built attack configuration (includes defense type).

    Raises:
        NotImplementedError: Raised when the defense is not implemented.

    Returns:
        Function: A defense instance to be executed.
    """
    if config.defense_type == "none":
        return None

    defense_func = DEFENSE_REGISTRY.get(config.defense_type)
    if defense_func is None:
        raise NotImplementedError(f"Action {config.defense_type} not implemented")

    return defense_func

def _train_and_measure(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_ds: Dataset,
    eval_ds: Dataset,
    config: AttackConfig,
    attack: Attack
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
        config (AttackConfig): User's built attack configuration.
        attack (Attack): User's attack object.

    Returns:
        tuple[PreTrainedModel, MetricSnapshot]: A tuple containing the new fine tuned model and its metrics.
    """
    trained_model = fine_tune(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        text_column=config.text_column,
        target_column=config.target_column,
        seed=config.seed,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        output_dir=config.output_dir,
        device=config.device
    )

    metrics = capture_metrics(
        model=trained_model,
        tokenizer=tokenizer,
        dataset=eval_ds,
        text_column=config.text_column,
        target_column=config.target_column,
        device=config.device
    )

    predictions = get_predictions(trained_model, tokenizer, eval_ds, config.text_column, config.device)
    metrics.attack_success_rate = attack.measure_success(predictions, eval_ds[config.target_column])

    return trained_model, metrics
    

def _split_dataset(dataset: Dataset, config: AttackConfig) -> tuple[Dataset, Dataset]:
    """
    Splits a given dataset based on attack configuration.

    Args:
        dataset (Dataset): A dataset to split.
        config (AttackConfig): User's defined attack configuration.

    Returns:
        tuple[Dataset, Dataset]: A tuple containing the train split and the eval split of the dataset.
    """
    split = dataset.train_test_split(test_size=config.test_size, seed=config.seed)
    train_ds, eval_ds = split["train"], split["test"]
    return train_ds, eval_ds


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
    _start = perf_counter()

    try:
        _attack = _build_attack(config)
        _defense = _build_defense(config)
    except (NotImplementedError, ValueError) as e:
        return RunResult(
            config=config,
            status="failed",
            error_message=str(e)
        )

    try:
        _dataset = _load_dataset(config)
    except (DatasetNotFoundError, FileNotFoundError, ValueError):
        return RunResult(
            config=config,
            status="failed",
            error_message=f"Could not find or access dataset '{config.dataset_name}'"
        )
    except IndexError:
        return RunResult(
            config=config,
            status="failed",
            error_message=f"Dataset '{config.dataset_name}' contains fewer than {config.samples} samples"
        )

    try:
        _model_base = _load_model(config.target_model)
        _tokenizer = _load_tokenizer(config.target_model)
    except (OSError, ValueError):
        return RunResult(
            config=config,
            status="failed",
            error_message=f"Could not load model or tokenizer '{config.target_model}'"
        )

    _train, _eval = _split_dataset(_dataset, config) # will add error catching to this and other stuff in a later dev stage!!!!11!

    try:
        _train_poisoned = _attack.apply(_train)
    except (NegevBaseError, ValueError) as e:
        return RunResult(
            config=config,
            status="failed",
            error_message=str(e)
        )

    # --- baseline model ---
    _, base_metrics = _train_and_measure(
        model=_model_base,
        tokenizer=_tokenizer,
        train_ds=_train,
        eval_ds=_eval,
        config=config,
        attack=_attack
    )

    # --- poisoned model ---
    _model_poisoned = _load_model(config.target_model)

    _, poisoned_metrics = _train_and_measure(
        model=_model_poisoned,
        tokenizer=_tokenizer,
        train_ds=_train_poisoned,
        eval_ds=_eval,
        config=config,
        attack=_attack
    )

    # --- sanitized model (only if configured) ---
    sanitized_metrics = None
    sanitized_report = None

    if config.defense_type != 'none':
        _train_sanitized, sanitized_report = _defense(
            dataset=_train_poisoned,
            model=_model_poisoned,
            tokenizer=_tokenizer,
            target_column=config.target_column,
            text_column=config.text_column,
            device=config.device
        )
        _model_sanitized = _load_model(config.target_model)
        
        _, sanitized_metrics = _train_and_measure(
            model=_model_sanitized,
            tokenizer=_tokenizer,
            train_ds=_train_sanitized,
            eval_ds=_eval,
            config=config,
            attack=_attack
        )

    elapsed_time = perf_counter() - _start

    return RunResult(
        config=config,
        status="complete",
        baseline_metrics=base_metrics,
        post_attack_metrics=poisoned_metrics,
        post_defense_metrics=sanitized_metrics,
        duration_seconds=elapsed_time,
        filter_report=sanitized_report
    )