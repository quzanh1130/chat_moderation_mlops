if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    x_train, y_train = data[0], data[1]
    x_validation, y_validation = data[2], data[3] 
    x_test, y_test = data[4], data[5]


    return  x_train, y_train, x_validation, y_validation, x_test, y_test

