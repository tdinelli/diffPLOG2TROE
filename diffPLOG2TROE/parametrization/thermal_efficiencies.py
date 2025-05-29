import equinox as eqx

from ..utilities.custom_types import Array64f_3, ScalarOrVector
from .arrhenius import Arrhenius


class ThermalEfficiency(eqx.Module):
    rate_like_expression: Arrhenius
    collider_name: str

    def __init__(self, parameters: Array64f_3, collider_name: str = "") -> None:
        self.rate_like_expression = Arrhenius(parameters=parameters)
        self.collider_name = collider_name

    @eqx.filter_jit
    def kinetic_constant(self, T: ScalarOrVector) -> ScalarOrVector:
        return self.rate_like_expression.kinetic_constant(T)
