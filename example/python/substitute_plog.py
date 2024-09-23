import jax.numpy as jnp
import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/")
from diffPLOG2TROE.pressure_logarithmic import kinetic_constant_plog
from diffPLOG2TROE.chemkin.chemkin_interpreter import read_chemkin_extract_plog
from diffPLOG2TROE.chemkin.chemkin_writer import write_chemkin
from diffPLOG2TROE.refitter import refit_plog

plog_reactions, idx_plog, idx_reactions = read_chemkin_extract_plog(
    # kinetics="/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/example/data/SMALL_WITH_DUPLICATE.CKI",
    # kinetics="/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/example/data/SMALL_WITH_DUPLICATE_AND_IMPLICIT.CKI",
    kinetics="/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/example/data/CRECK_C1C3_HT.CKI",
)


converted_plog_reactions = []
fitting_parameters = []
for i in plog_reactions:
    tmp = {"name": "", "parameters": [], "is_duplicate": False}
    if i["is_duplicate"] is True:
        tmp["is_duplicate"] = True

    print("Reaction: {}".format(i["name"]))

    jax_plog = jnp.asarray(i["parameters"], dtype=jnp.float64)

    A, b, Ea, R2adj, fg = refit_plog(jax_plog, 4.5)

    print(f"{A}, {b}, {Ea}, {R2adj}")

    tmp["name"] = i["name"]
    tmp["parameters"] = [float(A), float(b), float(Ea)]
    converted_plog_reactions.append(tmp)

    fg = jnp.append(fg, R2adj)
    fitting_parameters.append(fg)

write_chemkin(
    kinetics="/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/example/data/CRECK_C1C3_HT.CKI",
    output_folder="/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/example/python",
    plog_converted=converted_plog_reactions,
    fitting_parameters = fitting_parameters
)
