import pandas as pd

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to three Parquet files.

    Args:
        x_train, y_train, x_validation, y_validation, x_test, y_test: The output from the upstream parent block

    Output:
        None
    """
    print(data)

    # Combine x and y data into DataFrames
    train_df = pd.DataFrame({'x_train': data[0], 'y_train': data[1]})
    validation_df = pd.DataFrame({'x_validation': data[2], 'y_validation': data[3]})
    test_df = pd.DataFrame({'x_test': data[4], 'y_test': data[5]})

    # Export the data to Parquet files
    train_df.to_parquet('train_data.parquet')
    validation_df.to_parquet('validation_data.parquet')
    test_df.to_parquet('test_data.parquet')

