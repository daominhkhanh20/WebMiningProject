export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY='1a9353ffefccf6ac83c09c9e2f41959a'
kaggle datasets download -d daominhkhanh/bertwebmining -p assets/models/BertModel
cd assets/models/BertModel
unzip bertwebmining.zip
rm bertwebmining.zip