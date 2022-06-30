import argparse

from src.dataset import CNNDataSource
from src.learner import CNNLearner

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/_UIT-VSFC/csv')
parser.add_argument('--n_cnn', type=int, default=1)
parser.add_argument('--kernel_size', nargs="+", type=int, default=[3])
parser.add_argument('--pooling_kernel_size', type=int, default=2)
parser.add_argument('--out_channel', nargs="+", type=int, default=[32])
parser.add_argument('--n_dense', type=int, default=4)
parser.add_argument('--n_tensor_dense', nargs='+', type=int, default=[512, 256, 128, 64])
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--text_col', type=str, default='text')
parser.add_argument('--label_col', type=str, default='label')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--use_label_smoothing', default=False, type=lambda x: x.lower())
parser.add_argument('--smoothing_value', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--path_save_model', type=str, default='models')
parser.add_argument('--is_save_best_model', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--fine_tune', default=False, type=lambda x: x.lower() == 'true')
args = parser.parse_args()

datasource = CNNDataSource.init_datasource(path_folder_data=args.path_folder_data,
                                           max_length=args.max_length,
                                           text_col=args.text_col,
                                           label_col=args.label_col)

trainer = CNNLearner(
    mode='training',
    n_cnn=args.n_cnn,
    kernel_size=args.kernel_size,
    pooling_kernel_size=args.pooling_kernel_size,
    out_channel=args.out_channel,
    n_dense=args.n_dense,
    n_tensor_dense=args.n_tensor_dense,
    embedding_dim=args.embedding_dim,
    data_source=datasource,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    use_label_smoothing=args.use_label_smoothing,
    smoothing_value=args.smoothing_value,
    learning_rate=args.learning_rate,
    path_save_model=args.path_save_model,
    is_save_best_model=args.is_save_best_model,
    dropout=args.dropout
)

trainer.fit()
