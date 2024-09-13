import jax
from jax import lax
import jax.numpy as jnp

# Funzione per trovare l'indice con lax.fori_loop
def find_index(pIndex, i):
    # Se il valore corrente di P è minore o uguale al valore in _P[i]
    return lax.cond(
        P <= _P[i],
        lambda _: i,  # Restituisci l'indice corrente
        lambda _: pIndex,  # Mantieni l'indice corrente
        None
    )

# Simula il contesto in cui P e _P sono usati
P = 0.4
_P = jnp.array([0.1, 1.0, 10.0, 100.0])
n = len(_P)

# Usa lax.fori_loop per trovare l'indice corretto
pIndex = lax.fori_loop(0, n, find_index, 0)  # Inizializziamo pIndex a 0

print("Indice trovato:", pIndex)
