import argparse
from src.dataset import AnnDataSource
from src.learner import AnnLearner

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data')
parser.add_argument('--text_col', type=str, default='sentence')
parser.add_argument('--label_col', type=str, default='lb_name')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--path_save_model', type=str, default='assets/models')
parser.add_argument('--stopword_path', type=str,
                    default='assets/stopword/stopword.txt')
parser.add_argument('--path_save_tf', type=str,
                    default='assets/utils_weight')
parser.add_argument('--path_report', type=str,
                    default='assets/report')
parser.add_argument('--is_save_best_model', default=True)
parser.add_argument('--dropout', type=float, default=0.2)
args = parser.parse_args()

datasource = AnnDataSource.init_datasource(
    path_folder_data=args.path_folder_data,
    stopword_path=args.stopword_path,
    path_save_tf=args.path_save_tf,
    text_col=args.text_col,
    map_labels = {'negative':0,'neutral':1,'positive':2},
    label_col=args.label_col)

trainer = AnnLearner(
    mode='training',
    data_source=datasource,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    learning_rate=args.learning_rate,
    path_save_model=args.path_save_model,
    is_save_best_model=args.is_save_best_model,
    dropout=args.dropout,
    path_report=args.path_report
)

trainer.fit()
