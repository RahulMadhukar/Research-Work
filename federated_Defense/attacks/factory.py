from .label_flipping import StaticLabelFlipping, DynamicLabelFlipping
from .trigger import CentralizedTriggerAttack, CoordinatedTriggerAttack, RandomTriggerAttack, ModelDependentTriggerAttack
from .model_poisoning_attacks import (
    LocalModelReplacementAttack,
    LocalModelNoiseAttack,
    GlobalModelReplacementAttack,
    AggregationModificationAttack
)

class AttackFactory:
    @staticmethod
    def create_attack(config):
        # Label flipping attacks
        if config.attack_type == 'slf':
            return StaticLabelFlipping(config)
        elif config.attack_type == 'dlf':
            return DynamicLabelFlipping(config)

        # Trigger-based backdoor attacks
        elif config.attack_type == 'centralized':
            return CentralizedTriggerAttack(config)
        elif config.attack_type == 'coordinated':
            return CoordinatedTriggerAttack(config)
        elif config.attack_type == 'random':
            return RandomTriggerAttack(config)
        elif config.attack_type == 'model_dependent':
            return ModelDependentTriggerAttack(config)

        # Model poisoning attacks
        elif config.attack_type == 'local_model_replacement':
            return LocalModelReplacementAttack(config)
        elif config.attack_type == 'local_model_noise':
            return LocalModelNoiseAttack(config)
        elif config.attack_type == 'global_model_replacement':
            return GlobalModelReplacementAttack(config)
        elif config.attack_type == 'aggregation_modification':
            return AggregationModificationAttack(config)

        else:
            raise ValueError(f"Unknown attack type: {config.attack_type}")
