from tensorflow.keras.preprocessing import text, sequence

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype="float32")

def load_embeddings(file):
    with open(file) as f:
        return dict(get_coefs(*line.strip().split(" ")) for line in f)

def build_matrix(word_index, file):
    embedding_index = load_embeddings(file)
    embedding_matrix = np.zeros((len(word_index) + 1, 300), dtype=np.float32)
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            continue
    return embedding_matrix

@custom
def transform_custom(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Define constants
    MAX_LENGTH = 200
    BATCH_SIZE = 64
    EPOCHS = 15
    CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

    # Access the data using appropriate keys
    x_train = data.get('x_train')
    y_train = data.get('y_train')
    x_validation = data.get('x_validation')
    y_validation = data.get('y_validation')
    x_test = data.get('x_test')
    y_test = data.get('y_test')

    
    print(x_train)

    tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
    tokenizer.fit_on_texts(list(x_train) + list(x_validation) + list(x_test))

    # Convert texts to sequences
    x_train = tokenizer.texts_to_sequences(x_train)
    x_validation = tokenizer.texts_to_sequences(x_validation)
    x_test = tokenizer.texts_to_sequences(x_test)

    # Pad sequences to ensure consistent input lengths
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LENGTH)
    x_validation = sequence.pad_sequences(x_validation, maxlen=MAX_LENGTH)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LENGTH)

    return x_train, y_train, x_validation, y_validation, x_test, y_test

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
