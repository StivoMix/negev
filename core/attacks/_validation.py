from core.exceptions import InsufficientLabelsError, PoisonRateOutOfRange

def validate_poison_rate(poison_rate: float) -> None:
    if not 0 <= poison_rate <= 1:
        raise PoisonRateOutOfRange(poison_rate)


def validate_label_amount(unique_labels: list[str], required_label_amount: int, target_column: str) -> None:
    if len(unique_labels) < required_label_amount:
        raise InsufficientLabelsError(target_column, unique_labels)