tirex
================

.. currentmodule:: tirex

Load Model
-----------------

.. autofunction:: load_model
   :noindex:

Model interface
---------------

.. autoclass:: ForecastModel
   :members: max_context_length, forecast, forecast_gluon, forecast_hfdata

Utilities
---------

.. automodule:: tirex.util
   :members: select_quantile_subset
