import numpy as np
import pandas as pd

from tensorflow import keras

from benchmarks.common import create_shifted_frames, create_lagged_shifted_frames

from benchmarks.common import Dataset
from models.models import get_convarma, get_convlstm


def get_data(p):
    # Download and load the dataset.
    fpath = keras.utils.get_file(
        "moving_mnist.npy",
        "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
    )
    dataset = np.load(fpath)

    # Swap the axes representing the number of frames and number of data samples.
    dataset = np.swapaxes(dataset, 0, 1)

    # We'll pick out 1000 of the 10000 total examples and use those.
    dataset = dataset[:1000, ...]
    # Add a channel dimension since the images are grayscale.
    dataset = np.expand_dims(dataset, axis=-1)

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.seed(0)
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.45 * dataset.shape[0])]
    val_index = indexes[int(0.45 * dataset.shape[0]): int(0.5 * dataset.shape[0])]
    test_index = indexes[int(0.5 * dataset.shape[0]):]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]

    # Normalize the data to the 0-1 range.
    train_dataset = train_dataset / 255
    val_dataset = val_dataset / 255
    test_dataset = test_dataset / 255

    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)
    x_test, y_test = create_shifted_frames(test_dataset)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))
    print("Test Dataset Shapes: " + str(x_test.shape) + ", " + str(y_test.shape))
    bc = keras.losses.BinaryCrossentropy()
    print("Baseline: " + str(float(bc(y_test[:, p - 1:], x_test[:, p - 1:]))))

    X_train_lagged, Y_train = create_lagged_shifted_frames(train_dataset, p)
    X_val_lagged, Y_val = create_lagged_shifted_frames(val_dataset, p)
    X_test_lagged, Y_test = create_lagged_shifted_frames(test_dataset, p)

    arma_data = Dataset(X_train_lagged, Y_train, X_val_lagged, Y_val, X_test_lagged, Y_test)
    lstm_data = Dataset(x_train, y_train, x_val, y_val, x_test, y_test)

    return arma_data, lstm_data


def train_arma_model(arma_data: Dataset, layers: int, rep: int, dataset_name: str):
    arma_model = get_convarma(layers, arma_data.x_train.shape, arma_data.y_train.shape)
    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Define modifiable training hyperparameters.
    epochs = 20
    batch_size = 5

    # Fit the model to the training data.
    arma_model.fit(
        arma_data.x_train,
        arma_data.y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(arma_data.x_val, arma_data.y_val),
        callbacks=[early_stopping, reduce_lr],
    )
    file_name = f"{dataset_name}_ARMA_{layers}l_{rep}"
    model_path = file_name + ".hdf5"
    loss_path = file_name + ".csv"

    test_loss = arma_model.evaluate(arma_data.x_test, arma_data.y_test)
    print(f"{test_loss}, {layers}, {dataset_name}")

    arma_model.save(model_path)
    pd.Series({
        "Benchmark": dataset_name,
        "Layers": layers,
        "Repetition": rep,
        "Test_loss": test_loss
    }).to_csv(loss_path)


def train_lstm_model(lstm_data: Dataset, layers: int, rep: int, p: int, dataset_name: str):
    lstm_model = get_convlstm(layers, lstm_data.x_train.shape)
    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Define modifiable training hyperparameters.
    epochs = 20
    batch_size = 5

    # Fit the model to the training data.
    lstm_model.fit(
        lstm_data.x_train,
        lstm_data.y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(lstm_data.x_val, lstm_data.y_val),
        callbacks=[early_stopping, reduce_lr],
    )
    file_name = f"{dataset_name}_LSTM_{layers}l_{rep}"
    model_path = file_name + ".hdf5"
    loss_path = file_name + ".csv"
    test_loss_all = lstm_model.evaluate(lstm_data.x_test, lstm_data.y_test)

    bc = keras.losses.BinaryCrossentropy()
    lstm_pred = lstm_model.predict(lstm_data.x_test)
    test_loss = float(bc(lstm_data.y_test[:, p - 1:], lstm_pred[:, p - 1:]))
    print(f"{test_loss}, {layers}, {dataset_name}")

    lstm_model.save(model_path)
    pd.Series({
        "Benchmark": dataset_name,
        "Layers": layers,
        "Repetition": rep,
        "Test_loss": test_loss,
        "Test_loss_all": test_loss_all
    }).to_csv(loss_path)


def run_mnist(reps=1):
    p = 3
    dataset_name = "MNIST"

    print("#" * 40)
    print(dataset_name.center(40, "#"))
    print("#" * 40)

    arma_data, lstm_data = get_data(p)

    for rep in range(reps):

        print("#" * 40)
        print(f"{dataset_name} REP {rep}".center(40, "#"))
        print("#" * 40)

        for layers in [1, 2, 3]:
            print("#" * 40)
            print(f"{dataset_name} REP {rep} LAYERS {layers}".center(40, "#"))
            print("#" * 40)

            train_arma_model(arma_data, layers, rep, dataset_name)
            train_lstm_model(lstm_data, layers, rep, p, dataset_name)


if __name__ == '__main__':
    run_mnist()
