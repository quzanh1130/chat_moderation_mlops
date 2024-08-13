import tensorflow as tf

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here

    # Enable TensorFlow 1.x compatibility mode
    tf.compat.v1.disable_eager_execution()

    # Reset default graph
    tf.compat.v1.reset_default_graph()

    # Define placeholders
    inputs = tf.compat.v1.placeholder(shape=(None, MAX_LENGTH), dtype=tf.int32)
    keep_prob = tf.compat.v1.placeholder_with_default(1.0, shape=(), name='keep_prob')
    keep_prob_input = tf.compat.v1.placeholder_with_default(1.0, shape=(), name='keep_prob_input')
    keep_prob_conv = tf.compat.v1.placeholder_with_default(1.0, shape=(), name='keep_prob_conv')
    targets = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.float32)

    # Embedding layer
    embedding_layer = tf.Variable(embedding_matrix, trainable=False, name="embedding_layer")
    input_embeddings = tf.nn.embedding_lookup(embedding_layer, inputs)
    expanded_inputs = tf.expand_dims(input_embeddings, 3)
    expanded_inputs = tf.nn.dropout(expanded_inputs, rate=tf.cast(1.0 - keep_prob, dtype=expanded_inputs.dtype))

    # Convolutional layers
    conv_layer2 = tf.keras.layers.Conv2D(16, (2, 300), (1, 1), activation=tf.nn.relu)(expanded_inputs)
    conv_layer2 = tf.nn.dropout(conv_layer2, rate=tf.cast(1.0 - keep_prob_conv, dtype=conv_layer2.dtype))

    conv_layer4 = tf.keras.layers.Conv2D(16, (4, 300), (1, 1), activation=tf.nn.relu)(expanded_inputs)
    conv_layer4 = tf.nn.dropout(conv_layer4, rate=tf.cast(1.0 - keep_prob_conv, dtype=conv_layer4.dtype))

    conv_layer6 = tf.keras.layers.Conv2D(16, (6, 300), (1, 1), activation=tf.nn.relu)(expanded_inputs)
    conv_layer6 = tf.nn.dropout(conv_layer6, rate=tf.cast(1.0 - keep_prob_conv, dtype=conv_layer6.dtype))

    conv_layer8 = tf.keras.layers.Conv2D(16, (8, 300), (1, 1), activation=tf.nn.relu)(expanded_inputs)
    conv_layer8 = tf.nn.dropout(conv_layer8, rate=tf.cast(1.0 - keep_prob_conv, dtype=conv_layer8.dtype))

    # Squeeze and pooling layers
    squeeze2 = tf.squeeze(conv_layer2, 2)
    squeeze4 = tf.squeeze(conv_layer4, 2)
    squeeze6 = tf.squeeze(conv_layer6, 2)
    squeeze8 = tf.squeeze(conv_layer8, 2)

    pool2 = tf.keras.layers.MaxPooling1D(pool_size=MAX_LENGTH-2+1, strides=1)(squeeze2)
    pool4 = tf.keras.layers.MaxPooling1D(pool_size=MAX_LENGTH-4+1, strides=1)(squeeze4)
    pool6 = tf.keras.layers.MaxPooling1D(pool_size=MAX_LENGTH-6+1, strides=1)(squeeze6)
    pool8 = tf.keras.layers.MaxPooling1D(pool_size=MAX_LENGTH-8+1, strides=1)(squeeze8)

    # Concatenate pools
    pools = [pool2, pool4, pool6, pool8]
    pools = [tf.squeeze(x, 1) for x in pools]
    concat_layers = tf.concat(pools, axis=1)

    # Dense layers
    hidden_layer = tf.keras.layers.Dense(256, activation=tf.nn.relu)(concat_layers)
    final_layer = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(hidden_layer)

    # Define accuracy, optimizer, and loss
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(targets, 1), tf.argmax(final_layer, 1)), tf.float32))
    optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
    loss = tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=targets, logits=final_layer))

    var = tf.compat.v1.trainable_variables()
    lossl2 = tf.add_n([tf.nn.l2_loss(v) for v in var if 'bias' not in v.name and "embedding" not in v.name]) * 0.003

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    training_op = optimizer.minimize(loss + lossl2)
    training_op = tf.group([training_op, update_ops])

    
    return training_op


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
