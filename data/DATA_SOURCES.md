# U.S. Election Data Sources

**Last Updated:** 2025-11-20
**Used in:** `forecast_diffusion.py` diffusion forecast model

---

## Datasets

### 1. Presidential Results - State Level (1976-2020)
- **File:** `election_results/mit_president_state_1976_2020.csv`
- **Rows:** 4,288
- **Size:** 490 KB
- **Format:** Tab-separated values
- **Source:** MIT Election Data & Science Lab
- **URL:** https://dataverse.harvard.edu/api/access/datafile/4299753
- **Coverage:** Presidential elections 1976-2020, all 50 states + DC
- **Fields:** `year, state, state_po, state_fips, state_cen, state_ic, office, candidate, party_detailed, writein, candidatevotes, totalvotes, version, notes, party_simplified`
- **Usage:** Ground truth for 2016 election outcomes (model evaluation)

### 2. Individual Presidential Polls - 2016 Election (Time Series)
- **File:** `polls/fivethirtyeight_2016_polls_timeseries.csv`
- **Rows:** 4,209
- **Size:** 410 KB
- **Format:** CSV
- **Source:** FiveThirtyEight (aggregated from HuffPost Pollster, RealClearPolitics, polling firms, news reports)
- **URL:** https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/polls_us_election_2016.csv
- **Coverage:** Nov 2015 - Nov 2016, state-level and national polls throughout campaign
- **Fields:** `state, startdate, enddate, pollster, grade, samplesize, population, rawpoll_clinton, rawpoll_trump, rawpoll_johnson, rawpoll_mcmullin, adjpoll_clinton, adjpoll_trump, adjpoll_johnson, adjpoll_mcmullin`
- **Usage:** Time-series polling data for latent diffusion process and pollster bias estimation

---

## Citations

**MIT Election Data & Science Lab:**
```
MIT Election Data and Science Lab, 2017, "U.S. President 1976–2020",
https://doi.org/10.7910/DVN/42MVDX, Harvard Dataverse, V11
```

**FiveThirtyEight:**
```
FiveThirtyEight (2016), "Presidential Poll Data 2015-2016",
Aggregated from HuffPost Pollster, RealClearPolitics, and polling firms.
Available via R package dslabs and Rdatasets repository.
```
