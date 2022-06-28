
def make_input_bert(tokenizer, sent: str, max_length: int = 256):
    out = tokenizer(sent, max_length=max_length, truncation=True,
                    return_tensors="pt")
    return out['input_ids'].reshape(-1), out['attention_mask'].reshape(-1)


def make_input_cnn(tokenizer, sent: str):
    out = tokenizer.tokenize(sent)
    return out.reshape(-1)
