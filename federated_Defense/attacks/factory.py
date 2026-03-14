from .byzantine_attacks import GradientScalingAttack, SameValueAttack, BackGradientAttack

class AttackFactory:
    @staticmethod
    def create_attack(config):
        # Byzantine gradient manipulation attacks (CMFL paper Section 6.3.1)
        if config.attack_type == 'gradient_scaling':
            return GradientScalingAttack(config)
        elif config.attack_type == 'same_value':
            return SameValueAttack(config)
        elif config.attack_type == 'back_gradient':
            return BackGradientAttack(config)
        else:
            raise ValueError(f"Unknown attack type: {config.attack_type}")
