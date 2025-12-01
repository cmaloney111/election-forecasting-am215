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
     - Baseline
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
     - High
     - Best

1. Poll Average Model
----------------------

The simplest baseline model that computes a weighted average of recent polls.

**Key Features:**

* Uses 14-day polling window
* Weights polls by sample size
* Empirical uncertainty estimation
* Simple horizon adjustment for days until election

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

**Use Case:** Smooth time-series forecasting with trend

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

**Use Case:** More stable forecasts with better calibrated uncertainty

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

where :math:`h_p` is house effect for pollster :math:`p`, :math:`n_p` is number of polls, :math:`\lambda` is shrinkage parameter, and :math:`\bar{r}_p` is mean residual.

**File:** ``election_forecasting/models/hierarchical_bayes.py``

**Use Case:** Production-quality forecasts with best predictive accuracy

Model Comparison
----------------

All models are evaluated using three metrics:

1. **Brier Score:** Measures accuracy of probabilistic forecasts

   .. math::

      \text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2

   where :math:`p_i` is predicted win probability and :math:`y_i \in \{0,1\}` is actual outcome.

2. **Log Loss:** Cross-entropy between predictions and outcomes

   .. math::

      \text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N [y_i \log(p_i) + (1-y_i) \log(1-p_i)]

3. **MAE (Margin):** Mean absolute error of margin predictions

   .. math::

      \text{MAE} = \frac{1}{N} \sum_{i=1}^N |m_i^{\text{pred}} - m_i^{\text{actual}}|

Typical Performance (2016 Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the 2016 election data, the models typically achieve:

* **Hierarchical Bayes:** Brier ~0.08-0.12, MAE ~3-4%
* **Improved Kalman:** Brier ~0.10-0.14, MAE ~3-5%
* **Kalman Diffusion:** Brier ~0.12-0.16, MAE ~4-6%
* **Poll Average:** Brier ~0.14-0.18, MAE ~4-7%

(Results vary by forecast date)

Adding New Models
-----------------

To add a new forecasting model:

1. Create new file in ``election_forecasting/models/``
2. Inherit from ``ElectionForecastModel`` base class
3. Implement ``fit_and_forecast()`` method returning:

   * ``win_probability``: float in [0,1]
   * ``predicted_margin``: float (Democratic margin)
   * ``margin_std``: float (standard deviation)

4. Add model to ``run_all_models.py``

Example:

.. code-block:: python

   from election_forecasting.models.base_model import ElectionForecastModel

   class MyModel(ElectionForecastModel):
       def __init__(self):
           super().__init__("my_model")

       def fit_and_forecast(self, state_polls, forecast_date,
                           election_date, actual_margin):
           # Your forecasting logic here
           return {
               "win_probability": p,
               "predicted_margin": m,
               "margin_std": s,
           }
