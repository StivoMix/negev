from .label_noise_filter import label_noise_filter_apply

DEFENSE_REGISTRY = {
    "label_noise_filter": label_noise_filter_apply
}