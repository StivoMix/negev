from .base import Attack
from .label_flip import LabelFlipAttack
from .targeted_label_flip import TargetedLabelFlipAttack

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "label_flip": LabelFlipAttack,
    "targeted_label_flip": TargetedLabelFlipAttack
}