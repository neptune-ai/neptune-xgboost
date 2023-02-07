import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def dataset():
    # Prepare datasets
    data = load_iris()
    return train_test_split(data["data"], data["target"], test_size=0.2)
