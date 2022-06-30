from src.learner import *
from src.utils.create_data import clean_text
from src.utils.io import get_config_architecture


class SentimentInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config_architecture = get_config_architecture(model_path)
        self.model_name = self.config_architecture['model_name']
        if self.model_name == 'BertLearner':
            self.learner = BertLearner(
                mode='inference',
                path_save_model=model_path
            )
        else:
            raise Exception(f"{self.model_name} isn't support")

    def predict(self, sample: str):
        sample = clean_text(sample)
        log_predict = self.learner.predict(sample)
        return log_predict

