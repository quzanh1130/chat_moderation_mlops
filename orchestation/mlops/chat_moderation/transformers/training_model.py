import mlflow
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from IPython.display import clear_output
import mlflow.tensorflow

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(training_op, tokenize_data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    TRACKING_SERVER_HOST = "127.0.0.1" # Changing it with the IP address of the mlflow server
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

    # Assuming the necessary variables and placeholders are already defined
    # inputs, keep_prob, keep_prob_input, keep_prob_conv, targets, training_op, loss, accuracy, final_layer, EPOCHS, BATCH_SIZE, x_train, y_train, x_validation, y_validation, x_test, y_test

    saver = tf.compat.v1.train.Saver()

    # Lists to store loss and accuracy values
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    save_path = "models/"


    EXPERIMENT_NAME = "chat_removal"

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start an MLflow run
    with mlflow.start_run():
        mlflow.tensorflow.autolog()  # Automatically log TensorFlow metrics and parameters

        # Log model parameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(EPOCHS):
                avg_train_error = []
                avg_train_acc = []
                train_acc = 0
                t0 = time.time()
                for i in range((len(x_train) // BATCH_SIZE)):
                    x_batch = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                    y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                    feed = {inputs: x_batch, keep_prob: 0.5, keep_prob_input: 0.9, keep_prob_conv: 0.5, targets: y_batch}
                    _, train_loss, train_acc = sess.run([training_op, loss, accuracy], feed)
                    avg_train_error.append(train_loss)
                    avg_train_acc.append(train_acc)

                clear_output(wait=True)
                feed = {inputs: x_validation, targets: y_validation}
                validation_err, validation_acc = sess.run([loss, accuracy], feed)            
                shuffled_indices = np.random.choice(x_train.shape[0], x_train.shape[0], False)
                x_train = x_train[shuffled_indices]
                y_train = y_train[shuffled_indices]
                
                # Store the average training loss and accuracy
                avg_train_loss = sum(avg_train_error) / len(avg_train_error)
                avg_train_accuracy = sum(avg_train_acc) / len(avg_train_acc)
                train_losses.append(avg_train_loss)
                train_accuracies.append(avg_train_accuracy)
                
                # Store the validation loss and accuracy
                validation_losses.append(validation_err)
                validation_accuracies.append(validation_acc)
                
                print("Epoch: " + str(epoch + 1))
                print("Average error: {:.5f}".format(avg_train_loss))
                print("Average accuracy: {:.5f}".format(avg_train_accuracy))
                print("Validation error: {:.5f}".format(validation_err))
                print("Validation accuracy: {:.5f}".format(validation_acc))
                
                # Log the average training and validation metrics for each epoch
                mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("avg_train_accuracy", avg_train_accuracy, step=epoch)
                mlflow.log_metric("validation_loss", validation_err, step=epoch)
                mlflow.log_metric("validation_accuracy", validation_acc, step=epoch)

            clear_output(wait=True)
            feed = {inputs: x_test, targets: y_test}
            test_loss, test_acc, pred = sess.run([loss, accuracy, final_layer], feed)
            auc_score = roc_auc_score(y_test[:,1], pred[:,1])
            print("Test error: {:.5f}".format(test_loss))
            print("Test accuracy: {:.5f}".format(test_acc))
            print("AUC score: {:.5f}".format(auc_score))

            # Log the final metrics
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("auc_score", auc_score)

            # Save the model
            saver.save(sess, save_path)
            print(f"Model saved in path: {save_path}")

            # Log the model checkpoint as an artifact
            mlflow.log_artifact(save_path)
            # Get the artifact URI
            artifact_uri = mlflow.get_artifact_uri("model")
            print(f"Artifact URI: {artifact_uri}")

            # Register the model with the MLflow model registry
            mlflow.register_model(
                artifact_uri,
                "MyModel"
            )

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'