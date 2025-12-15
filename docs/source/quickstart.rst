Quick Start Guide
=================

Basic Usage
-----------

Arrhenius Reactions
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from diffPLOG2TROE.kinetics import Arrhenius

   # Define Arrhenius parameters
   parameters = {
       "A": 1e13,    # Pre-exponential factor [1/s]
       "n": 0.0,     # Temperature exponent
       "Ea": 50000   # Activation energy [cal/mol]
   }

   # Create Arrhenius object
   reaction = Arrhenius(parameters, name="H2 + OH -> H2O + H")

   # Calculate rate constant at different temperatures
   import jax.numpy as jnp
   T = jnp.linspace(300, 2000, 100)  # Temperature range [K]
   k = reaction.rate_constant(T)

PLOG Reactions
~~~~~~~~~~~~~~

.. code-block:: python

   from diffPLOG2TROE.kinetics import Plog

   # Define PLOG parameters (pressure-dependent)
   plog_params = {
       0.1: {"A": 1e12, "n": 0.0, "Ea": 45000},    # 0.1 atm
       1.0: {"A": 1e13, "n": 0.0, "Ea": 50000},    # 1.0 atm
       10.0: {"A": 1e14, "n": 0.0, "Ea": 55000},   # 10.0 atm
   }

   # Create PLOG object
   plog_reaction = Plog(plog_params, name="pressure_dependent")

   # Calculate rate constants
   T = 1000.0  # Temperature [K]
   P = jnp.logspace(-1, 2, 50)  # Pressure range [atm]
   k = plog_reaction.rate_constant(T, P)

Falloff Reactions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from diffPLOG2TROE.kinetics import FallOff

   # High pressure limit parameters
   hpl_params = {"A": 1e14, "n": 0.0, "Ea": 0}

   # Low pressure limit parameters  
   lpl_params = {"A": 1e16, "n": -1.0, "Ea": 0}

   # Troe parameters
   troe_params = {
       "A": 0.5,      # Troe parameter
       "T3": 1000,    # T*** [K]
       "T1": 100,     # T* [K] 
       "T2": 5000     # T** [K]
   }

   # Create falloff reaction
   falloff = FallOff(
       hpl_params, 
       lpl_params, 
       "troe", 
       troe_params,
       name="falloff_example"
   )

   # Calculate rate constants
   T = jnp.linspace(300, 2000, 100)
   P = jnp.logspace(-2, 2, 50)
   k = falloff.rate_constant(T, P)
