Quick Start Guide
=================

This guide will help you get started with the election forecasting package.

Running All Models
------------------

The easiest way to get started is to run the complete pipeline:

.. code-block:: bash

   election-run-all

This will:

1. Run all four forecasting models on default forecast dates
2. Generate model comparison metrics and plots
3. Create state-level visualizations for key swing states

Custom Forecast Dates
----------------------

To run forecasts on a different number of dates:

.. code-block:: bash

   election-run-all --dates 8

Individual Commands
-------------------

Run Forecasts
~~~~~~~~~~~~~

.. code-block:: bash

   # Run with default 4 forecast dates
   election-forecast

   # Run with custom number of dates
   election-forecast --dates 6

   # Run with verbose output
   election-forecast -v

Compare Models
~~~~~~~~~~~~~~

After running forecasts, compare model performance:

.. code-block:: bash

   election-compare

This generates:

* ``model_comparison.csv`` - Detailed metrics table
* ``model_comparison.png`` - Performance visualization
* Console output with rankings

Generate Plots
~~~~~~~~~~~~~~

Create state-level forecast plots:

.. code-block:: bash

   # Plot key swing states (default)
   election-plot

   # Plot all states with polling data
   election-plot --all

   # Plot specific states
   election-plot --states FL PA MI WI

Using Models Programmatically
------------------------------

You can also use the models directly in Python:

.. code-block:: python

   from election_forecasting.models.hierarchical_bayes import HierarchicalBayesModel
   import pandas as pd

   # Initialize model
   model = HierarchicalBayesModel()

   # Run forecast
   forecast_dates = [pd.to_datetime("2016-10-15"), pd.to_datetime("2016-11-01")]
   predictions = model.run_forecast(forecast_dates=forecast_dates, verbose=True)

   # Save results
   metrics = model.save_results()
   print(metrics)

   # Plot specific state
   model.plot_state("FL")

Understanding the Output
-------------------------

Predictions Directory
~~~~~~~~~~~~~~~~~~~~~

``predictions/`` contains CSV files with model predictions:

* ``state``: Two-letter state code
* ``forecast_date``: Date forecast was made
* ``win_probability``: Probability of Democratic win (0-1)
* ``predicted_margin``: Predicted Democratic margin
* ``margin_std``: Standard deviation of margin
* ``actual_margin``: Actual election result

Metrics Directory
~~~~~~~~~~~~~~~~~

``metrics/`` contains text files with evaluation metrics:

* **Brier Score**: Accuracy of probabilistic forecasts (lower is better)
* **Log Loss**: Cross-entropy loss (lower is better)
* **MAE**: Mean absolute error of margin predictions (lower is better)

Plots Directory
~~~~~~~~~~~~~~~

``plots/`` contains PNG visualizations organized by model, showing:

* Raw polling data (gray points)
* Model forecasts over time (blue line)
* 90% confidence intervals (light blue bands)
* Actual election result (red dashed line)
