Forecasting Models
==================

This package implements four different election forecasting models, each with different approaches to handling polling data.

Model Overview
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Model
     - Approach
     - Complexity
     - Performance
   * - Poll Average
     - Weighted average of recent polls
     - Low
     - Baseline (yet second best)
   * - Kalman Diffusion
     - Brownian motion with Kalman filter
     - Medium
     - Good
   * - Improved Kalman
     - Kalman filter with drift and regularization
     - Medium
     - Better
   * - Hierarchical Bayes
     - Bayesian ensemble with bias correction
     - Highest
     - Best

1. Poll Average Model
----------------------

The simplest baseline model that computes a weighted average of recent polls.

**Key Features:**

* Uses 14-day polling window
* Weights polls by sample size
* Empirical uncertainty estimation
* Simple horizon adjustment for days until election (decrease uncertainty as we approach election)

**File:** ``election_forecasting/models/poll_average.py``

**Use Case:** Quick baseline forecast, minimal computational cost

2. Kalman Diffusion Model
--------------------------

Implements a Brownian motion model using Kalman filter with Rauch-Tung-Striebel (RTS) smoother.

**Key Features:**

* Brownian motion with drift dynamics
* Forward Kalman filter + backward RTS smoother
* EM algorithm for parameter estimation
* Handles irregular time spacing in polls

**Mathematical Model:**

.. math::

   x_{t+1} = x_t + \mu \cdot dt + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2 \cdot dt)

   y_t = x_t + v_t, \quad v_t \sim N(0, R_t)

where :math:`\mu` is drift, :math:`\sigma^2` is diffusion variance, and :math:`R_t` is observation noise.

**File:** ``election_forecasting/models/kalman_diffusion.py``

3. Improved Kalman Model
-------------------------

Enhanced version of Kalman Diffusion with stronger regularization and better uncertainty quantification.

**Key Features:**

* Increased minimum diffusion variance
* Regularized pollster bias estimates
* Conservative probability clipping
* Reduced forecast horizon uncertainty

**Improvements over Basic Kalman:**

* Minimum diffusion variance: :math:`\sigma^2_{\min} = 0.003^2`
* Pollster bias regularization parameter: :math:`\lambda = 5.0`
* Probability clipping: [0.02, 0.98] instead of [0.01, 0.99]

**File:** ``election_forecasting/models/improved_kalman.py``

4. Hierarchical Bayes Model (Best)
-----------------------------------

The most sophisticated model combining multiple information sources with systematic bias correction.

**Key Features:**

1. **Fundamentals Prior:** Weighted average of 2012 (70%) and 2008 (30%) election results
2. **Kalman-Filtered Polls:** RTS smoother on recent polling data
3. **House Effects:** Hierarchical shrinkage estimation of pollster biases
4. **Systematic Bias Correction:** Adaptive correction for polling errors
5. **Proper Uncertainty Quantification:** Combines multiple uncertainty sources

**Model Components:**

.. math::

   \text{Forecast} = w_{\text{prior}} \cdot \mu_{\text{fund}} + w_{\text{polls}} \cdot \mu_{\text{polls}} - \text{bias}

   \text{Total Variance} = \sigma^2_{\text{combined}} + \sigma^2_{\text{evolution}} + \sigma^2_{\text{bias}}

**House Effects Estimation:**

.. math::

   h_p = \frac{n_p}{n_p + \lambda} \cdot \bar{r}_p

where :math:`h_p` is house effect for pollster :math:`p`, :math:`n_p` is number of polls, :math:`\lambda` is shrinkage parameter, and :math:`\bar{r}_p` is mean residual. Essentially, if a pollster tends to have very different results compared to other pollsters in a similar time period, shrink their effect.

**File:** ``election_forecasting/models/hierarchical_bayes.py``

Adding New Models
-----------------

We have made all scripts scalable so that all one needs to do to add a new forecasting mode is:

1. Create new file in ``election_forecasting/models/``
2. Inherit from ``ElectionForecastModel`` base class
3. Implement the ``fit_and_forecast()`` method, which should return:

   * ``win_probability``: float in [0,1]
   * ``predicted_margin``: float (Democratic margin)
   * ``margin_std``: float (standard deviation)

Example:

.. code-block:: python

   from election_forecasting.models.base_model import ElectionForecastModel

   class NewModel(ElectionForecastModel):
       def __init__(self):
           super().__init__("new_model")

       def fit_and_forecast(self, state_polls, forecast_date,
                           election_date, actual_margin):
           # Your forecasting logic here
           return {
               "win_probability": p,
               "predicted_margin": m,
               "margin_std": s,
           }
