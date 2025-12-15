# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .gbm_regressor import TirexGBMRegressor
from .linear_regressor import TirexLinearRegressor
from .rf_regressor import TirexRFRegressor

__all__ = ["TirexLinearRegressor", "TirexRFRegressor", "TirexGBMRegressor"]
