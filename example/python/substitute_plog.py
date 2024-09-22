import jax.numpy as jnp
import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/")
from diffPLOG2TROE.pressure_logarithmic import kinetic_constant_plog
from diffPLOG2TROE.chemkin.read_chemkin import read_chemkin_extract_plog
from diffPLOG2TROE.refitter import refit_plog

plog_reactions, idx_plog, idx_reactions = read_chemkin_extract_plog(
    kinetics="/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/example/data/C1C3_NOx_221230.CKI.EXT"
)


for i in plog_reactions:
    print("Reaction: {}".format(i["name"]))
    jax_plog = jnp.asarray(i["parameters"], dtype=jnp.float64)
    A, b, Ea, R2adj = refit_plog(jax_plog, 1.)
    print("  * Adjusted R2: {:.3}".format(R2adj))
    print("    - A: {:.3e}, b: {:.3}, Ea: {:.3e}".format(A, b, Ea))
