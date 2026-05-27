"""
Configures data poisoning and adversarial attack configurations and runs.

This module provides Pydantic models to define adversarial machine learning
attacks, track model performance metrics, and log benchmark run execution data.
"""

from pydantic import BaseModel, Field, computed_field
from typing import Literal
from datetime import datetime
import uuid


class AttackConfig(BaseModel):
    """
    A class representing an attack configuration. Includes attack type and target.

    Attributes:
        attack_type (Literal): The type of the attack to run on a model
        target_model (str): The target model to run the attack on
        poison_rate (float): Percentage of the dataset that'll be corrupted. 10% by default, min 0% and max 100%
        dataset_name (str): The ID of the dataset in huggingface
        notes (str): Notes written by the attacker to document testing
        defense_type (Literal): The type of defense to use on a poisoned dataset. Defaults to "none".
        text_column (str): The column in the dataset containing the predictable sequences.
        target_column (str): The column containing the labels to be predicted by a model.
        seed (int): An RNG seed for reproducibility. Defaults to 0.
        epochs (int): How many times should a model iterate over a datset. Defaults to 3.
        samples (int | None): How many rows to use in model fine tuning. Defaults to None.
        device (str): Which device to use for model fine tuning and computing. Defaults to "cpu".
    """
    attack_type: Literal[
        "label_flip",
        "targeted_label_flip",
        "data_injection",
        "data_churn",
        "backdoor_trigger",
        "clean_label_backdoor",
        "lora_weight_poison",
        "gradient_manipulation",
        "feature_collision",
    ]
    target_model: str
    poison_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    dataset_name: str
    text_column: str
    target_column: str
    notes: str = ""
    defense_type: Literal[
        "none",
        "label_noise_filter",
    ] = "none"
    seed: int = 0
    epochs: int = 3
    samples: int | None = None      # None = use full dataset
    device: str = "cpu"



class MetricSnapshot(BaseModel):
    """
    A snapshot of a single point in time measurement of model performance

    Attributes:
        accuracy (float): The accuracy of the model
        perplexity (float): The perplexity of the model
        attack_success_rate (float): Measurement of how much impact the attack had on the model at the given point in time
    """
    accuracy: float
    perplexity: float | None = None
    attack_success_rate: float | None = None


def _generate_short_id() -> str:
    """
    Simple function that generates a short ID using uuid4()

    Returns:
        str: A short ID of 8 chars
    """
    return str(uuid.uuid4())[:8]
    

class FilterReport(BaseModel):
    """
    A post defense report holding data on suspicious rows in a dataset.

    Attributes:
        total_rows (int): Total rows scanned in a dataset.
        rows_flagged (int): Total rows that were flagged as suspicious in a dataset.
        method (str): Method used to filter for suspicious rows.
    """
    total_rows: int
    rows_flagged: int
    method: Literal[
        "label_noise_filter",
        "isolation_forest",
        "lof",
        "embedding_cluster"
    ] # more will be added in the future, this is a placeholder for now

    @computed_field
    @property
    def rows_remaining(self) -> int:
        """
        Calculates remaining, non flagged rows in a dataset after a scan.

        Returns:
            int: Non flagged rows in a dataset post scan.
        """
        return self.total_rows - self.rows_flagged
    

class RunResult(BaseModel):
    """
    The full result of a whole run

    Attributes:
        run_id (str): A randomly generated ID for the benchmark run
        config (AttackConfig): The config of the attack used for the run
        status (Literal): The status of the run
        baseline_metrics (MetricSnapshot): Initial metrics of a model (pre attack)
        post_attack_metrics (MetricSnapshot): Post attack metrics of a model
        post_defense_metrics (MetricSnapshot): Post defense metrics of a model
        created_at (datetime): Date at which the attack was created
        duration_seconds (float | None): Duration of the attack in seconds. Defaults to None.
        error_message (str | None): An error message to present incase of an exception. Defaults to None.
        filter_report (FilterReport | None): A post defense report. Defaults to None.
    """
    run_id: str = Field(default_factory=_generate_short_id)
    config: AttackConfig
    status: Literal["pending", "running", "complete", "failed"] = "pending"
    baseline_metrics: MetricSnapshot | None = None
    post_attack_metrics: MetricSnapshot | None = None
    post_defense_metrics: MetricSnapshot | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    duration_seconds: float | None = None
    error_message: str | None = None
    filter_report: FilterReport | None = None

    @computed_field
    @property
    def degradation(self) -> float | None:
        """
        Method calculates accuracy dropped post attack

        Returns:
            float: Accuracy dropped post attack
        """
        if self.baseline_metrics and self.post_attack_metrics:
            return self.baseline_metrics.accuracy - self.post_attack_metrics.accuracy
        
        return None