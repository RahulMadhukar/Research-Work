from .label_flipping import StaticLabelFlipping, DynamicLabelFlipping
from .trigger import CentralizedTriggerAttack, CoordinatedTriggerAttack, RandomTriggerAttack, ModelDependentTriggerAttack
from .model_poisoning_attacks import (
    LocalModelReplacementAttack,
    LocalModelNoiseAttack,
    GlobalModelReplacementAttack,
    AggregationModificationAttack
)
from .byzantine_attacks import GradientScalingAttack, SameValueAttack, BackGradientAttack

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

        # Byzantine gradient manipulation attacks
        elif config.attack_type == 'gradient_scaling':
            return GradientScalingAttack(config)
        elif config.attack_type == 'same_value':
            return SameValueAttack(config)
        elif config.attack_type == 'back_gradient':
            return BackGradientAttack(config)

        else:
            raise ValueError(f"Unknown attack type: {config.attack_type}")
