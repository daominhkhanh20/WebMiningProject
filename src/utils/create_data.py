import re
import gensim
import unicodedata


def make_input_bert(tokenizer, sent: str, max_length: int = 256):
    out = tokenizer(sent, max_length=max_length, truncation=True,
                    return_tensors="pt")
    return out['input_ids'].reshape(-1), out['attention_mask'].reshape(-1)


def make_input_cnn(tokenizer, sent: str):
    out = tokenizer.tokenize(sent)
    return out.reshape(-1)


def clean_text(sent):
    sent = unicodedata.normalize('NFC', sent)
    emoji = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002500-\U00002BEF"  # chinese char
                       u"\U00002702-\U000027B0"
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001f926-\U0001f937"
                       u"\U00010000-\U0010ffff"
                       u"\u2640-\u2642"
                       u"\u2600-\u2B55"
                       u"\u200d"
                       u"\u23cf"
                       u"\u23e9"
                       u"\u231a"
                       u"\ufe0f"  # dingbats
                       u"\u3030"
                       "]+", re.UNICODE)
    sent = re.sub(emoji, '', sent)
    return " ".join(gensim.utils.simple_preprocess(sent))
