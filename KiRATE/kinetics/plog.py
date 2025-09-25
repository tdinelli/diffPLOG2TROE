import re
from typing import Dict, List, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float64, Int64

from KiRATE.kinetics.arrhenius import Arrhenius


class Plog(eqx.Module):
    _k_levels: List[Arrhenius]
    _p_levels: Float64[Array, "np"]
    _lnp_levels: Float64[Array, "np"]
    _num_p_levels: Int64[Array, ""]
    _k0: Optional[Arrhenius] = None
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        parameters: Dict[float, Dict[str, float]],
        name: str = "",
        k0: Optional[Dict[str, float]] = None,
    ) -> None:
        self._name = name

        # ==============================================================================
        # Sort pressure levels in ascending order
        parameters = dict(sorted(parameters.items()))

        self._p_levels = jnp.array(list(parameters.keys()), dtype=jnp.float64)
        self._lnp_levels = jnp.log(self._p_levels)
        self._num_p_levels = jnp.int64(len(self._p_levels))

        # Create Arrhenius objects and store them in a JAX array
        arrhenius_objects = []
        for p, params in parameters.items():
            arrhenius_objects.append(Arrhenius(parameters=params, name=f"{name} ({p})"))

        self._k_levels = arrhenius_objects

        if k0 is not None:
            self._k0 = Arrhenius(parameters=k0, name=f"{name} (k0)")
        else:
            self._k0 = None

    @classmethod
    def from_chemkin(cls, input_string: str) -> Union["Plog", Tuple["Plog", "Plog"]]:
        """
        Create Plog instance(s) from a CHEMKIN format string.

        This class method provides a convenient way to construct Plog objects
        directly from CHEMKIN-style input strings, which are commonly used in
        chemical kinetics databases and modeling software.

        Parameters
        ----------
        input_string : str
            CHEMKIN-formatted string containing reaction name and PLOG entries.
            Expected format:
            ```
            REACTION_NAME    A0   n0   Ea0
            PLOG / pressure1 A1  n1   Ea1 /
            PLOG / pressure2 A2  n2   Ea2 /
            ...
            ```

            For duplicate reactions:
            ```
            REACTION_NAME    A0   n0   Ea0
            DUPLICATE
            PLOG / pressure1 A1  n1   Ea1 /
            PLOG / pressure1 A1' n1'  Ea1'/
            PLOG / pressure2 A2  n2   Ea2 /
            PLOG / pressure2 A2' n2'  Ea2'/
            ```

        Returns
        -------
        Union[Plog, Tuple[Plog, Plog]]
            - Single Plog instance for standard case (no duplicate pressures)
            - Tuple of two Plog instances for duplicate case (implicit duplicates)

        Raises
        ------
        ValueError
            If the input string cannot be parsed or contains invalid parameters.

        See Also
        --------
        parse_chemkin_entry : Static method used internally for parsing
        """
        # Parse the CHEMKIN entry using the existing static method
        parsed_result = cls.parse_chemkin_entry(input_string)

        if len(parsed_result) == 3:
            # Standard case: no duplicates
            reaction_name, k0_params, plog_coefficients = parsed_result

            # Convert coefficient lists to parameter dictionaries
            parameters = {}
            for pressure, coeffs in plog_coefficients.items():
                parameters[pressure] = {"A": coeffs[0], "n": coeffs[1], "Ea": coeffs[2]}

            return cls(parameters=parameters, name=reaction_name, k0=k0_params)
        elif len(parsed_result) == 4:
            # Duplicate case: implicit duplicates found
            reaction_name, k0_params, plog_coefficients_1, plog_coefficients_2 = parsed_result

            parameters_1 = {}
            for pressure, coeffs in plog_coefficients_1.items():
                parameters_1[pressure] = {"A": coeffs[0], "n": coeffs[1], "Ea": coeffs[2]}

            parameters_2 = {}
            for pressure, coeffs in plog_coefficients_2.items():
                parameters_2[pressure] = {"A": coeffs[0], "n": coeffs[1], "Ea": coeffs[2]}

            plog_1 = cls(parameters=parameters_1, name=reaction_name, k0=k0_params)
            plog_2 = cls(parameters=parameters_2, name=reaction_name, k0=k0_params)
            return plog_1, plog_2
        else:
            raise ValueError(f"Unexpected number of return values from parse_chemkin_entry: {len(parsed_result)}")

    @eqx.filter_jit
    def rate_constant(
        self,
        T: Union[float, Float64[Array, ""], Float64[Array, "nt"]],
        P: Union[float, Float64[Array, ""], Float64[Array, "np"]],
    ) -> Union[Float64[Array, ""], Float64[Array, "nt"], Float64[Array, "np"], Float64[Array, "nt np"]]:
        """Compute kinetic constant for given temperature and pressure."""
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)

        if jnp.isscalar(P):  # P is scalar
            return self._single_P_rate_constant(T, P)
        else:  # P is array
            vec_func = vmap(lambda p: self._single_P_rate_constant(T, p))
            return vec_func(P)

    def _single_P_rate_constant(
        self,
        T: Union[Float64[Array, ""], Float64[Array, "nt"]],
        P: Float64[Array, ""],
    ) -> Union[Float64[Array, ""], Float64[Array, "nt"]]:
        all_lnk = jnp.log(jnp.array([k_level.rate_constant(T) for k_level in self.k_levels]))

        # ==============================================================================
        # Identify the region of the table
        is_below_min = P <= self._p_levels[0]
        is_above_max = P >= self._p_levels[-1]

        k = lax.cond(
            is_below_min,
            lambda _: all_lnk[0],
            lambda _: lax.cond(
                is_above_max,
                lambda _: all_lnk[-1],
                lambda _: self._interpolated_constant(all_lnk, P),
                None,
            ),
            None,
        )
        return jnp.exp(k)

    def _interpolated_constant(
        self,
        all_lnk: Union[Float64[Array, "np"], Float64[Array, "nt np"]],
        P: Float64[Array, ""],
    ) -> Union[Float64[Array, ""], Float64[Array, "nt"]]:
        # TODO: Check the typing definition

        # ==============================================================================
        # Log-log interpolation for pressures within range
        upper_idx = self._find_index(P)  # Position of the current pressure value in the pressure levels of the plog
        lower_idx = upper_idx - 1

        upper_lnp = self._lnp_levels[upper_idx]
        lower_lnp = self._lnp_levels[lower_idx]

        upper_lnk = all_lnk[upper_idx]
        lower_lnk = all_lnk[lower_idx]

        return self._log_log_interpolation(lower_lnk, upper_lnk, lower_lnp, upper_lnp, P)

    def _find_index(self, P: Float64[Array, ""]) -> Int64[Array, ""]:
        # TODO: Check the typing definition

        # ==============================================================================
        # Get the first insertion point where P <= p_levels[i]
        indices = jnp.searchsorted(self._p_levels, P, side="left")

        # ==============================================================================
        # If P is greater than all values in p_levels, set index to the last element
        indices = jnp.where(indices == self._num_p_levels, self._num_p_levels - 1, indices)

        return indices

    @staticmethod
    def _log_log_interpolation(
        log_k1: Union[Float64[Array, ""], Float64[Array, "nt"]],
        log_k2: Union[Float64[Array, ""], Float64[Array, "nt"]],
        log_P1: Float64[Array, ""],
        log_P2: Float64[Array, ""],
        P: Float64[Array, ""],
    ) -> Union[Float64[Array, ""], Float64[Array, "nt"]]:
        # TODO: Check the typing definition
        return log_k1 + (log_k2 - log_k1) * (jnp.log(P) - log_P1) / (log_P2 - log_P1)

    # ==================================================================================
    # CHEMKIN string parser
    @staticmethod
    def parse_chemkin_entry(input_string: str):
        lines = input_string.strip().split("\n")
        if not lines:
            raise ValueError("Empty CHEMKIN representation")

        main_line = lines[0].strip()
        if not main_line:
            raise ValueError("First line must contain reaction equation")

        # Extract reaction name using the same logic as Arrhenius.parse_chemkin_entry
        reaction_name, k0_params = Arrhenius.parse_chemkin_entry(main_line)

        # Compile regex pattern once
        pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

        # Single pass through data
        plog_coefficients_1 = {}
        plog_coefficients_2 = {}
        has_duplicates = False

        for line in lines[1:]:
            line = line.strip()

            if not line or "DUP" in line or "DUPLICATE" in line:
                continue

            tokens = line.split("/")[1:]
            if len(tokens) < 2:
                raise ValueError(f"Invalid PLOG format: {line}")

            plog_line = tokens[0].strip()
            try:
                plog_coefficients = [float(x) for x in pattern.findall(plog_line)]
            except ValueError:
                raise ValueError(f"Could not parse PLOG line {line}")

            pressure = plog_coefficients[0]
            coeffs = plog_coefficients[1:]

            # Direct assignment based on whether pressure already exists
            if pressure in plog_coefficients_1:
                # This is a duplicate
                has_duplicates = True
                if pressure not in plog_coefficients_2:
                    plog_coefficients_2[pressure] = coeffs
                else:
                    # More than 2 duplicates - convert to list if needed
                    if not isinstance(plog_coefficients_2[pressure][0], list):
                        plog_coefficients_2[pressure] = [plog_coefficients_2[pressure]]
                    plog_coefficients_2[pressure].append(coeffs)
            else:
                plog_coefficients_1[pressure] = coeffs

        if has_duplicates:
            return reaction_name, k0_params, plog_coefficients_1, plog_coefficients_2
        else:
            return reaction_name, k0_params, plog_coefficients_1

    # ==================================================================================
    # String Representations and Debugging
    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        if self._k0 is not None:
            str_obj = f"{self.name}\t\t{self._k0.A:.5E} {self._k0.n:.5E} {self._k0.Ea:.5E}\n"
        else:
            str_obj = f"{self.name}\t\t{0.0:.5E} {0.0:.5E} {0.0:.5E}\n"

        for i in range(self._num_p_levels):
            arrhenius = self._k_levels[i]
            str_obj += f" PLOG / {self._p_levels[i]:.5E}\t{arrhenius.A:.5E} {arrhenius.n:.5E} {arrhenius.Ea:.5E} /\n"
        return str_obj

    def __repr__(self) -> str:
        representer_string = "Plog(\n"
        representer_string += f" name   = {self._name},\n"
        representer_string += f" P      = {self._p_levels},\n"
        representer_string += f" nP     = {float(self._num_p_levels)},\n"
        representer_string += " params = [\n"
        for i in range(self._num_p_levels):
            level = self._k_levels[i]
            representer_string += f"  Arrhenius(\n"
            representer_string += f"   name = {level.name}\n"
            representer_string += f"   A    = {level.A}\n"
            representer_string += f"   n    = {level.n}\n"
            representer_string += f"   Ea   = {level.Ea}\n"
            representer_string += f"  ),\n"

        representer_string += " ]\n)"

        return representer_string

    # ==================================================================================
    # Properties for parameters access
    @property
    def name(self) -> str:
        return self._name

    @property
    def p_levels(self) -> Float64[Array, "np"]:
        return self._p_levels

    @property
    def lnp_levels(self) -> Float64[Array, "np"]:
        return self._lnp_levels

    @property
    def num_p_levels(self) -> Int64[Array, ""]:
        return self._num_p_levels

    @property
    def k_levels(self) -> List[Arrhenius]:
        return self._k_levels

    @property
    def k0(self) -> Optional[Arrhenius]:
        if self._k0 is not None:
            return self._k0
        else:
            return None
