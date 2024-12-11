[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pkharyuk/yellowcab_ridge/HEAD?labpath=.%2Fdemo.ipynb)

# **Predicting demand for NY taxi with Ridge Regression: Demo**

## About

The current repository contains a demonstration of the forecasting the demand for New York taxi
by using the ridge regression model (predictions for June 2016).
The following feature space is used:
- periodic functions (sines and cosines) expected to capture hourly details within a week ($T = 2 \pi/168$, $168 = 24 \cdot 7$);
- one-hot encoded seasonality feature (three 0-1 features, encoding seasons as $000$, $100$, $010$ and $001$);
- their pairwise interactions followed by feature selection by thresholding the coefficients);
- additional features such as average ride distance per hour, average number of passengers per hour, average ride duration, and average cost;
    - separate for every of $102$ regions;
    - logarithmically scaled, $x \to \log(1+x)$"
    
Link to the data:
- [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

## Install dependencies

This repository is distributed with the environment.yml file suitable for conda/mamba
package manager. It is highly recommended to use the mambaforge/micromamba:
- [https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)


Create environment from .yml file:
```
mamba env create environment.yml
```

Remove environment:
```
mamba env remove -n yellowcab_ridge
```

Activate environment and run Jupyter lab server:
```
mamba activate yellowcab_ridge
jupiterlab
```
