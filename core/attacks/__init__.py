from .base import Attack
from .label_flip import LabelFlipAttack
from .targeted_label_flip import TargetedLabelFlipAttack

ATTACK_REGISTRY = {
    "label_flip": {
        "class": LabelFlipAttack,
        "extra_fields": [],
    },
    "targeted_label_flip": {
        "class": TargetedLabelFlipAttack,
        "extra_fields": [
            {"name": "source_label", "type": "string", "required": True},
            {"name": "target_label", "type": "string", "required": True},
        ],
    },
}