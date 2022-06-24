from src.dataset import BertDataSource, CommentCollate
from torch.utils.data import DataLoader
datasource = BertDataSource.init_datasource(
    path_folder_data='assets/data',
    pretrained_model_name='vinai/phobert-base',
    max_length=256,
    text_col='comment',
    label_col='pred_label'
)


val_loader = DataLoader(datasource.val_dataset, batch_size=4, shuffle=False, collate_fn=CommentCollate(pad_id=1))

for idx, sample in enumerate(val_loader):
    try:
        label = sample['labels']
    except:
        pass