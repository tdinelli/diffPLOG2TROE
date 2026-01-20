import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import Threebody
from tests.test_utils import assert_rate_constants_close


class TestThreeBody(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)

        self.reaction = Threebody(
            name="OH+H+M=OH+M",
            parameters={"A": 5.000e17, "n": -1.000, "Ea": 0.000},
            efficiencies={"H2": 2.000, "H2O": 6.00, "CH4": 2.00, "CO": 1.50, "CO2": 2.00, "C2H6": 3.00, "AR": 0.70},
        )

        # ==============================================================================
        # Dataloader
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "3body.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate = jnp.array(data[:, 1]) * 1000 * 1000

    def test_kinetic_constant_n2(self):
        """Test three-body k0 rate constant (Cantera convention)."""
        calculated_rates = self.reaction.k0.rate_constant(T=self.T_range)

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate,
            T_range=self.T_range,
            test_name="Three-body k0 rate constant (N2 bath gas)",
        )


if __name__ == "__main__":
    unittest.main()
