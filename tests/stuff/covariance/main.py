import jax
import jax.numpy as jnp
from parameter_covariance import main as analyze_covariance

# Import from your main script
from diffPLOG2TROE.kinetic_constants import FallOff, Plog


def main():
    T_range = jnp.linspace(500, 2500, 100)
    P_range = jnp.logspace(jnp.log10(0.1), jnp.log10(100), 100)
    plog_dictionary = {
        "name": "NH2+H=NH3",
        "type": "plog",
        "rate-constant": {
            "coefficients": [
                [1.00000e-01, 1.21300e27, -4.95900e00, 2.80709e03],
                [1.00000e00, 5.86700e27, -4.86700e00, 3.10796e03],
                [1.00000e01, 3.31400e28, -4.79800e00, 3.83238e03],
                [1.00000e02, 4.51200e28, -4.56300e00, 4.72319e03],
            ]
        },
    }
    starting_plog = Plog(plog_dictionary)
    k_plog = starting_plog.kinetic_constant(T_range, P_range)

    # Define the loss function
    def ciao(params):
        falloff = FallOff(
            name="",
            hpl_params=jnp.array([1.500e14, 0.167, 0.000e00]),
            lpl_params=jnp.array([jnp.exp(params[0]), params[1], params[2] * 1.987]),
            falloff_params=jnp.array([params[3], params[4], params[5], params[6]]),
            falloff_type="troe",
        )
        predictions = falloff.kinetic_constant(T_range, P_range)[0]
        return jnp.mean((jnp.log(predictions) - jnp.log(k_plog)) ** 2)

    # Load your optimized parameters (replace with your actual optimal parameters)
    # This is just an example - use your actual optimized parameters from the previous run
    optimal_params = jnp.array(
        [jnp.log(9.06953e+30), -4.19661, 3.22327e+03 / 1.987, 6.04311e-01, 2.79973e+03, 4.43770e+01, 6.82702e+03],
        dtype=jnp.float64,
    )

    # Define parameter names for reference
    param_names = ["ln(A) [cm³/mol·s]", "n", "E/R [K]", "α", "T*** [K]", "T* [K]", "T** [K]"]

    # Analyze covariance
    covariance_info = analyze_covariance(ciao, optimal_params, param_names)

    return covariance_info


if __name__ == "__main__":
    # Set JAX to use double precision
    jax.config.update("jax_enable_x64", True)

    # Run the analysis
    covariance_info = main()
