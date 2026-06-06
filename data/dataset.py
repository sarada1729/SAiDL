import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class LanguageModelingDataset(Dataset):
    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length
        self.chunk_len = context_length + 1

    def __len__(self):
        return max(0, len(self.token_ids) - self.chunk_len)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.chunk_len]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_dataloaders(context_length=128, batch_size=4, tokenizer_name="gpt2"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_text = "\n\n".join(dataset["train"]["text"])
    valid_text = "\n\n".join(dataset["validation"]["text"])
    test_text = "\n\n".join(dataset["test"]["text"])

    train_ids = tokenizer(train_text, add_special_tokens=False)["input_ids"]
    valid_ids = tokenizer(valid_text, add_special_tokens=False)["input_ids"]
    test_ids = tokenizer(test_text, add_special_tokens=False)["input_ids"]

    train_dataset = LanguageModelingDataset(train_ids, context_length)
    valid_dataset = LanguageModelingDataset(valid_ids, context_length)
    test_dataset = LanguageModelingDataset(test_ids, context_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, tokenizer