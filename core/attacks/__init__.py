from .base import Attack
from .label_flip import LabelFlipAttack

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "label_flip": LabelFlipAttack
}