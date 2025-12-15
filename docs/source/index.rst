KiRATE Documentation
===========================

KiRATE gas

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/modules
   examples
   contributing

Installation
============

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/tdinelli/diffPLOG2TROE
   cd diffPLOG2TROE
   pip install .

Quick Start
===========

.. code-block:: python

   from KiRATE.kinetics import Arrhenius

   # Create an Arrhenius reaction
   params = {"A": 1e13, "n": 0.0, "Ea": 50000}
   reaction = Arrhenius(params, name="example")

   # Calculate rate constant at 1000 K
   k = reaction.rate_constant(1000.0)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
