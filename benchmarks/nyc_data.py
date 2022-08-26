import numpy as np
from tensorflow import keras

from benchmarks.common import create_sequences, scale, create_shifted_frames, create_lagged_shifted_frames, Dataset
from benchmarks.moving_mnist import train_arma_model, train_lstm_model


def get_data(p, dataset_name):
    sequence_length = 20

    train = np.load(f"benchmarks/data/NY{dataset_name}/{dataset_name.lower()}_train.npz")["flow"]
    test = np.load(f"benchmarks/data/NY{dataset_name}/{dataset_name.lower()}_test.npz")["flow"]
    val = np.load(f"benchmarks/data/NY{dataset_name}/{dataset_name.lower()}_val.npz")["flow"]

    train = np.log(train + 1)
    test = np.log(test + 1)
    val = np.log(val + 1)

    train_sequences = create_sequences(train, sequence_length)
    val_sequences = create_sequences(val, sequence_length)
    test_sequences = create_sequences(test, sequence_length)

    min_val, max_val = np.min(train_sequences), np.max(train_sequences)

    train_sequences = scale(test_sequences, min_val, max_val)
    val_sequences = scale(val_sequences, min_val, max_val)
    test_sequences = scale(test_sequences, min_val, max_val)

    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_sequences)
    x_val, y_val = create_shifted_frames(val_sequences)
    x_test, y_test = create_shifted_frames(test_sequences)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))
    print("Test Dataset Shapes: " + str(x_test.shape) + ", " + str(y_test.shape))
    bc = keras.losses.BinaryCrossentropy()
    print("Baseline: " + str(float(bc(y_test[:, p - 1:], x_test[:, p - 1:]))))

    X_train_lagged, Y_train = create_lagged_shifted_frames(train_sequences, p)
    X_val_lagged, Y_val = create_lagged_shifted_frames(val_sequences, p)
    X_test_lagged, Y_test = create_lagged_shifted_frames(test_sequences, p)

    arma_data = Dataset(X_train_lagged, Y_train, X_val_lagged, Y_val, X_test_lagged, Y_test)
    lstm_data = Dataset(x_train, y_train, x_val, y_val, x_test, y_test)

    return arma_data, lstm_data


def run_nyc(reps=1):
    p = 3

    for dataset_name in ["Taxi", "Bike"]:

        print("#" * 40)
        print(dataset_name.center(40, "#"))
        print("#" * 40)

        arma_data, lstm_data = get_data(p, dataset_name)

        for rep in reversed(list(range(1, reps))):

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
    run_nyc()
