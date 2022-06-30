from src.inference import SentimentInference

inference = SentimentInference(
    model_path='assets/models/BertModel'
)

print(inference.predict("Cô dạy khá hay"))

"""
Result:
 {'pred_label': 'positive', 
 'probability': [0.9999295473098755, 5.33430738869356e-06, 6.507167563540861e-05]}
"""
