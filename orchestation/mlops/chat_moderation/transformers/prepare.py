import numpy as np
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform(data_test, data_train, *args, **kwargs):
    """
    Template code for a transformer block.
    """
    assert isinstance(data_test, pd.DataFrame), 'data_test should be a DataFrame'
    assert isinstance(data_train, pd.DataFrame), 'data_train should be a DataFrame'

    x_col = "body"
    y_col = "REMOVED"

    train_df = data_train
    test_df = data_test

    # Define the size of the validation set
    val_size = 2000

    # Shuffle the indices
    shuffled_indices = np.random.permutation(len(train_df))

    # Split the shuffled indices for validation and training sets
    val_indices = shuffled_indices[-val_size:]
    train_indices = shuffled_indices[:-val_size]

    # Create the validation and training DataFrames
    val_df = train_df.iloc[val_indices].reset_index(drop=True)

    x_train = train_df[x_col].values
    y_train = train_df[y_col].values.astype(np.int32)
    onehot_train = np.zeros((y_train.size, y_train.max() + 1))
    onehot_train[np.arange(y_train.size), y_train] = 1
    y_train = onehot_train

    x_test = test_df[x_col].values
    y_test = test_df[y_col].values.astype(np.int32)
    onehot_test = np.zeros((y_test.size, y_test.max() + 1))
    onehot_test[np.arange(y_test.size), y_test] = 1
    y_test = onehot_test

    x_validation = val_df[x_col].values
    y_validation = val_df[y_col].values.astype(np.int32)
    onehot_val = np.zeros((y_validation.size, y_validation.max() + 1))
    onehot_val[np.arange(y_validation.size), y_validation] = 1
    y_validation = onehot_val

    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    return x_train, y_train, x_validation, y_validation, x_test, y_test

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert len(output) == 6, 'Output should contain 6 elements: x_train, y_train, x_validation, y_validation, x_test, y_test'
