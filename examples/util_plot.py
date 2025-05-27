from matplotlib import pyplot as plt


def plot_fc(ctx, quantile_fc, real_future_values=None):
    median_forecast = quantile_fc[:, 4].numpy()
    lower_bound = quantile_fc[:, 0].numpy()
    upper_bound = quantile_fc[:, 8].numpy()

    original_x = range(len(ctx))
    forecast_x = range(len(ctx), len(ctx) + len(median_forecast))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(original_x, ctx, label="Ground Truth Context", color="#4a90d9")
    if real_future_values is not None:
        original_fut_x = range(len(ctx), len(ctx) + len(real_future_values))
        plt.plot(original_fut_x, real_future_values, label="Ground Truth Future", color="#4a90d9", linestyle=":")
    plt.plot(forecast_x, median_forecast, label="Forecast (Median)", color="#d94e4e", linestyle="--")
    plt.fill_between(
        forecast_x, lower_bound, upper_bound, color="#d94e4e", alpha=0.1, label="Forecast 10% - 90% Quantiles"
    )
    plt.xlim(left=0)
    plt.legend()
    plt.grid(True)
    plt.show()
