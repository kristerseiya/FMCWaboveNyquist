
## Official Repository for A. Ulvog, J. Rapp, and V. Goyal. "FMCW Lidar Beyond Nyquist by Instantaneous Frequency Fitting", IEEE Transactions on Instrumentation and Measurement 2025.

## Dependencies
Python 3.10.14, numpy, scipy, matplotlib, 

## Tutorial

We use **FMCWMeasurement** object to generate a measurement.

```python
fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
t = np.arange(0, 2*T*n_cycle, 1./sample_rate)
dist, vel = 100, 10
signal, second_output = fmcw_meas.generate(dist, vel, t)
```

**FMCWMeasurement** takes **FMCWMeasurementProperties** object as an input which defines the configurations of the measurement such as bandwidth, chirp duration, type of frequency modulation, carrier wavelength, linewidth, etc.

## Running Demo

demo/ directory contains scripts that will generate a measurment and run estimators. Estimators can be defined in the following way.
```python
## Proposed estimator
key = {'gridgen_type':'optimal', 'method':'BFGS', 'init_step': 'none', 'ignore_quadrature':False, 'snr_adjustment':True}
estimator1 = estimators.IFRegressor(meas_prop, **key)

# Maximum Periodogram
estimator2 = estimators.MaximumPeriodogram(meas_prop)

# # Lorentzian Fitting
estimator3 = estimators.LorentzianRegressor(meas_prop, method="BFGS")

# Matched Filter
estimator4 = estimators.MathedFilterDelayEstimator(meas_prop, optimize=True)

# Tsuchida's Method
estimator5 = estimators.FrequencyAveraging(meas_prop)
```

For running a demo script for joint distance-velocity estimation, just simply run
```bash
python demo/dv_est.py
```