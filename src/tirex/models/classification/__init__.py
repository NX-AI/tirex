# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .heads.gbm_classifier import TirexGBMClassifier
from .heads.linear_classifier import TirexClassifierTorch
from .heads.rf_classifier import TirexRFClassifier

__all__ = ["TirexClassifierTorch", "TirexRFClassifier", "TirexGBMClassifier"]
