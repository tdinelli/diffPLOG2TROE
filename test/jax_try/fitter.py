import jax
import jax.numpy as jnp
import jax.scipy.optimize

# Le tue liste di input
k = jnp.array([1.09E-33, 1.80E-18, 4.66E-11, 9.97E-07, 6.43E-04, 5.75E-02, 1.52E+00, 1.80E+01, 1.23E+02, 5.63E+02,
               1.92E+03, 5.25E+03, 1.21E+04])

T = jnp.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])

# Funzione ArrheniusFitter già scritta
def arrhenius_loss(params, T_range, ln_k0):
    ln_k0_fit = params[0] + params[1] * jnp.log(T_range) - params[2] / 1.987 / T_range
    return jnp.sum((ln_k0 - ln_k0_fit) ** 2)

def ArrheniusFitter(k: jnp.ndarray, T_range: jnp.ndarray):
    ln_k0 = jnp.log(k)

    # Parametri iniziali per l'ottimizzazione
    initial_guess = jnp.array([ln_k0[0], 0.0, 10000.0])

    # Minimizza la funzione obiettivo
    result = jax.scipy.optimize.minimize(arrhenius_loss, initial_guess, args=(T_range, ln_k0), method='BFGS')

    popt = result.x

    # Estrai i parametri
    A = jnp.exp(popt[0])
    b = popt[1]
    Ea = popt[2]

    # Calcolo dell'R2
    ln_k_fit = popt[0] + b * jnp.log(T_range) - Ea / 1.987 / T_range
    R2 = 1 - jnp.sum((ln_k0 - ln_k_fit) ** 2) / jnp.sum((ln_k0 - jnp.mean(ln_k0)) ** 2)

    # Calcolo dell'R2 aggiustato
    R2adj = 1 - (1 - R2) * (len(T_range) - 1) / (len(T_range) - 1 - 2)

    # Controllo della precisione flottante per A
    A = jax.lax.cond(jnp.less_equal(A, jnp.finfo(float).eps) or jnp.isclose(A, 0),
                     lambda: jnp.inf,
                     lambda: A)

    return A, b, Ea, R2adj

# Esegui il fitting con le tue liste
A, b, Ea, R2adj = ArrheniusFitter(k, T)

# Stampa i risultati
print(f"A = {A}")
print(f"b = {b}")
print(f"Ea = {Ea}")
print(f"R² aggiustato = {R2adj}")
