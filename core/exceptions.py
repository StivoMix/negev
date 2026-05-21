class NegevBaseError(Exception):
    """Base exception for all errors raised by the Negev tool."""
    pass


class InsufficientLabelsError(NegevBaseError):
    """Raised when a dataset lacks enough unique labels for a label flip attack."""
    def __init__(self, target_column: str, found_labels: list):
        self.target_column = target_column
        self.found_labels = found_labels
        self.message = (
            f"Cannot perform label flipping on column '{target_column}'. "
            f"Label flipping requires at least 2 unique classes, but found "
            f"{len(found_labels)}: {found_labels}"
        )
        super().__init__(self.message)


class PoisonRateOutOfRange(NegevBaseError):
    """Raised when a poison rate exceeds its bounds of 0.0 to 1.0"""
    def __init__(self, poison_rate: float):
        self.poison_rate = poison_rate
        self.message = (
            f"Poison rate must be in [0.0, 1.0]. Recieved {poison_rate}."
        )
        super().__init__(self.message)