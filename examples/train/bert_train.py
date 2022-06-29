import argparse

from src.dataset import BertDataSource
from src.learner import BertLearner

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data')
parser.add_argument('--pretrained_model_name', type=str, default='vinai/phobert-base')
parser.add_argument('--max_length', type=int, default=256)
parser.add_argument('--text_col', type=str, default='comment')
parser.add_argument('--label_col', type=str, default='pred_label')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--use_label_smoothing', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--smoothing_value', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--path_save_model', type=str, default='assets/models')
parser.add_argument('--is_save_best_model', default=True, type=lambda x: x.lower() == 'true')
parser.add_argument('--mode_increase_weight_neural', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--neural_weight', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--fine_tune', default=True, type=lambda x: x.lower() == 'true')
args = parser.parse_args()

datasource = BertDataSource.init_datasource(
    path_folder_data=args.path_folder_data,
    pretrained_model_name=args.pretrained_model_name,
    max_length=args.max_length,
    text_col=args.text_col,
    label_col=args.label_col
)

trainer = BertLearner(
    mode='training',
    pretrained_model=args.pretrained_model_name,
    data_source=datasource,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    use_label_smoothing=args.use_label_smoothing,
    smoothing_value=args.smoothing_value,
    learning_rate=args.learning_rate,
    path_save_model=args.path_save_model,
    is_save_best_model=args.is_save_best_model,
    dropout=args.dropout,
    fine_tune=args.fine_tune
)

trainer.fit()