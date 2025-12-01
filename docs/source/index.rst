Election Forecasting Documentation
===================================

State-level presidential election forecasting using polling time-series data from the 2016 U.S. presidential election.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   models
   api
   development

Features
--------

* Four different forecasting models (Poll Average, Kalman Diffusion, Improved Kalman, Hierarchical Bayes)
* Comprehensive evaluation metrics (Brier Score, Log Loss, MAE)
* State-level time-series visualization
* Automated model comparison pipeline

Installation
------------

.. code-block:: bash

   pip install election-forecasting

Or with uv:

.. code-block:: bash

   uv pip install election-forecasting

Quick Start
-----------

.. code-block:: bash

   # Run complete pipeline: forecast, compare, and plot
   election-run-all

   # Run individual commands
   election-forecast
   election-compare
   election-plot

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
