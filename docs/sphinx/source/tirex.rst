tirex
================

.. currentmodule:: tirex

Load Model
-----------------

.. autofunction:: load_model
   :noindex:

Forecasting Model interface
---------------------------

.. autoclass:: ForecastModel
   :members: max_context_length, forecast, forecast_gluon, forecast_hfdata

Utilities
---------

.. _module-tirex.util:

.. automodule:: tirex.util
   :members:
      select_quantile_subset,
      plot_forecast
