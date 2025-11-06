from app.config import Settings

from tirex import ForecastModel, load_model


class TirexModel:
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        self.model: ForecastModel = load_model(
            settings.model_path,
            device=settings.model_device,
            backend="torch",
            compile=settings.model_compile,
        )

    def warmup(self) -> None:
        if self.settings.model_compile:
            print("Compile the model. That might take over 2 minutes...")
            _, __ = self.model.forecast(context=[list(range(2048))], prediction_length=32)
            print("Compilation done.")

    def predict(self, context, prediction_length):
        return self.model.forecast(context=context, prediction_length=prediction_length)
