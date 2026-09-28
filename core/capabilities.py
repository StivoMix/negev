from core.attacks import ATTACK_REGISTRY
from core.defenses import DEFENSE_REGISTRY

def get_capabilities() -> dict:
    """
    Returns all current tool capabilities including implemented attacks and defenses.

    Returns:
        dict: A dictionary containing implemented attacks and defenses.
    """
    return {
        "attacks": {
            name: {k: v for k, v in metadata.items() if k != "class"}
            for name, metadata in ATTACK_REGISTRY.items()
        },
        "defenses": ["none"] + list(DEFENSE_REGISTRY.keys()),
    }