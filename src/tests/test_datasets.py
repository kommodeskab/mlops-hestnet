import torch
from src.datasets import DummyDataset
from src.datasets import DGigawordDataset
from src.datasets.utils import get_tokenize_function

SEED = 42
torch.manual_seed(SEED)


# here is an example test function
def test_dummy_dataset():
    size = 10
    dataset = DummyDataset(size=size)
    assert len(dataset) == size, f"Dataset length should be {size}"
    for i in range(size):
        data = dataset[i]
        assert "input" in data
        assert "target" in data
        assert data["input"].shape == (10,), "Input shape should be (10,)"
        assert data["target"].shape == (1,), "Target shape should be (1,)"


def test_gigaword_dataset():
    dataset = DGigawordDataset()
    N_TRAIN = 471550
    idx = (torch.randperm(N_TRAIN)[:100]).tolist()
    idx = [0] + idx + [-1]
    assert len(dataset) == N_TRAIN, f"Dataset length should be {N_TRAIN}"

    checkpoint = "distilbert/distilgpt2"
    tokenize_function = get_tokenize_function(checkpoint)

    for i in idx:
        data = tokenize_function(dataset[i])
        assert "input_ids" in data, "Data dictionary should contain 'input_ids' key"
        assert "attention_mask" in data, "Data dictionary should contain 'attention_mask' key"
        assert data["input_ids"].ndim == 2, "input_ids shape should be (1, x)"
        assert data["input_ids"].shape[0] == 1 and data["input_ids"].shape[1] <= 1024, (
            f"input_ids shape should be (1, <=1024), got {data['input_ids'].shape}"
        )

        assert data["input_ids"].ndim == 2, "input_ids shape should be (1, x)"
        assert data["input_ids"].shape[0] == 1 and data["input_ids"].shape[1] <= 1024, (
            f"input_ids shape should be (1, <=1024), got {data['input_ids'].shape}"
        )
        assert data["input_ids"].shape == data["attention_mask"].shape, (
            f"input_ids and attention_mask shapes must match, got {data['input_ids'].shape} vs {data['attention_mask'].shape}"
        )
        assert data["attention_mask"].unique() == torch.tensor([1]), (
            f"attention_mask should only contain 1s, got {data['attention_mask'].unique()}"
        )


if __name__ == "__main__":
    test_dummy_dataset()
    test_gigaword_dataset()
