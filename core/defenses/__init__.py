from .label_noise_filter import LabelNoiseFilter
from .base import Defense

DEFENSE_REGISTRY = {
    "label_noise_filter": LabelNoiseFilter
}