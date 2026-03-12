
import numpy as np

from fmcw_sys import FMCWMeasurementProperties

class Estimator():
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        pass

    def estimate(self, t: np.ndarray, mixed_signal: np.ndarray, second_output: np.ndarray|None=None):
        raise NotImplementedError()

    def set_test_flag(self, test: bool):
        pass


class OracleEstimator():
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        pass

    def estimate(self, x: np.ndarray, t: np.ndarray, mixed_signal: np.ndarray, second_output: np.ndarray|None=None):
        raise NotImplementedError()
    
    def set_test_flag(self, test: bool):
        pass