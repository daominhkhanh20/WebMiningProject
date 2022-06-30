from typing import Union
from fastapi import FastAPI
from src.learner import AnnLearner
import json
from src.inference import SentimentInference
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def init_ann():
    f = open('assets/models/ann_config_architecture.json')
    cnf = json.load(f)
    global ann_trainer 
    ann_trainer = AnnLearner(
        mode='inference',
        infer_parm=cnf['model']
    )

init_ann()
bert_inference = SentimentInference(
    model_path='assets/models/BertModel'
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict/")
def return_result(model: str, q: Union[str, None] = None):
    if model == 'ANN':
        if q == None:
            return 'None'
        else:
            result = ann_trainer.predict(q)
            return result
    elif model == 'Bert':
        if q is None:
            return None
        else:
            log = bert_inference.predict(q)
            return log['pred_label']