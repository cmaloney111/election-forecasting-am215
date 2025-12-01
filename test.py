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