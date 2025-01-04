diffRateConstant
============================================

This documentation covers the differentiable Rate Constant package, which
provides tools for parsing and handling different types of chemical reaction
rate constants.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Features
--------

* Support for multiple reaction types:
    * Arrhenius reactions
    * Pressure-dependent logarithmic (PLOG) reactions
    * Falloff reactions
    * Chemically Activated Bimolecular Reactions (CABR)
    * Three-body reactions

* Flexible parameter parsing
* JAX integration for numerical operations
* Comprehensive error checking and validation

Installation
------------

To install the package, run:

.. code-block:: bash

   pip install reaction-kinetics

Getting Started
-----------------

Here's a simple example of parsing an Arrhenius reaction rate constant:

.. code-block:: python

   from reaction_kinetics import parse_rate_constant

   # Define an Arrhenius rate constant
   rate_constant = {
       "type": "Arrhenius",
       "parameters": [1e13, 0, 50000]  # [A, n, E]
   }

   # Parse the rate constant
   params = parse_rate_constant(rate_constant)

Indices and tables
=====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
