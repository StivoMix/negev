from pydantic import BaseModel, Field
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
    notes: str = ""


class MetricSnapshot(BaseModel):
    """
    A snapshot of a single point in time measurement of model performance

    Attributes:
        accuracy (float): The accuracy of the model
        perplexity (float): The perplexity of the model
        attack_success_rate (float): Measurement of how much impact the attack had on the model at the given point in time
    """
    accuracy: float
    perplexity: float
    attack_success_rate: float | None = None


def generate_short_id() -> str:
    """
    Simple function that generates a short ID using uuid4()

    Returns:
        str: A short ID of 8 chars
    """
    return str(uuid.uuid4())[:8]


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
        duration_seconds (float): Duration of the attack in seconds
    """
    run_id: str = Field(default_factory=generate_short_id)
    config: AttackConfig
    status: Literal["pending", "running", "complete", "failed"] = "pending"
    baseline_metrics: MetricSnapshot | None = None
    post_attack_metrics: MetricSnapshot | None = None
    post_defense_metrics: MetricSnapshot | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    duration_seconds: float | None = None

    @property
    def degradation(self) -> float | None:
        """
        Function calculates accuracy dropped post attack

        Returns:
            float: Accuracy dropped post attack
        """
        if self.baseline_metrics and self.post_attack_metrics:
            return self.baseline_metrics.accuracy - self.post_attack_metrics.accuracy
        
        return None