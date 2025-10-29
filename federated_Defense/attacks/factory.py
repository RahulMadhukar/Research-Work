from .label_flipping import StaticLabelFlipping, DynamicLabelFlipping
from .trigger import CentralizedTriggerAttack, CoordinatedTriggerAttack, RandomTriggerAttack, ModelDependentTriggerAttack

class AttackFactory:
    @staticmethod
    def create_attack(config):
        if config.attack_type == 'slf':
            return StaticLabelFlipping(config)
        elif config.attack_type == 'dlf':
            return DynamicLabelFlipping(config)
        elif config.attack_type == 'centralized':
            return CentralizedTriggerAttack(config)
        elif config.attack_type == 'coordinated':
            return CoordinatedTriggerAttack(config)
        elif config.attack_type == 'random':
            return RandomTriggerAttack(config)
        elif config.attack_type == 'model_dependent':
            return ModelDependentTriggerAttack(config)
        else:
            raise ValueError(f"Unknown attack type: {config.attack_type}")
