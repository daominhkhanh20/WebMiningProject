from src.inference import SentimentInference
import unittest

"""
inference.predict("Cô dạy khá hay")

Result:
 {'pred_label': 'positive', 
 'probability': [0.9999295473098755, 5.33430738869356e-06, 6.507167563540861e-05]}
"""


class TestingModel(unittest.TestCase):
    def setUp(self):
        self.inference = SentimentInference(
            model_path='assets/models/BertModel'
        )

    def test_model_bert(self):
        test_case = {
            "Cô dạy khá hay": "positive",
            "bình thường": "neural",
            "abcxyz abcxyz": "neural",
            "cô hơi già": "negative"
        }
        for sent, label in test_case.items():
            log = self.inference.predict(sent)
            self.assertEqual(log['pred_label'], label)
