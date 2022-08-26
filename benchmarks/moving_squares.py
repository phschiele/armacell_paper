import numpy as np

from tensorflow import keras

from benchmarks.moving_mnist import train_arma_model, train_lstm_model
from benchmarks.common import create_shifted_frames, create_lagged_shifted_frames, Dataset


def generate_movies(n_samples=1200, n_frames=20):
    row = 80
    col = 80
    shifted_movies = np.zeros((n_samples, n_frames + 1, row, col, 1),
                              dtype=np.float)
    larger_noisy = np.zeros((n_samples, n_frames + 1, row, col, 1),
                            dtype=np.float)
    np.random.seed(0)
    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)
        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(1, 8) * np.sign(np.random.rand() - 0.5)
            directiony = np.random.randint(1, 8) * np.sign(np.random.rand() - 0.5)
            # Size of the square
            w = np.random.randint(4, 8)
            shifted_movies[i, 0, xstart - w: xstart + w, ystart - w: ystart + w, 0] += 1
            larger_noisy[i, 0, xstart - w - 1: xstart + w + 1, ystart - w - 1: ystart + w + 1, 0] += 1
            for t in range(1, n_frames + 1):
                # Shift the ground truth by 1
                shifted_movies[i, t] = np.roll(shifted_movies[i, 0], int(directionx * (t)), axis=-2)
                shifted_movies[i, t] = np.roll(shifted_movies[i, t], int(directiony * (t)), axis=-3)
                larger_noisy[i, t] = np.roll(larger_noisy[i, 0], int(directionx * (t)), axis=-2)
                larger_noisy[i, t] = np.random.rand() * np.roll(larger_noisy[i, t], int(directiony * (t)), axis=-3)
    noisy_movies = np.maximum(shifted_movies, larger_noisy)
    # Cut to a 40x40 window
    noisy_movies = noisy_movies[:, 1:, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[:, 1:, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies


def get_data(p, dataset_name):
    noisy_movies, shifted_movies = generate_movies(n_samples=1000)

    if dataset_name == "NOISY_MOVIES":
        dataset = noisy_movies
    elif dataset_name == "SHIFTED_MOVIES":
        dataset = shifted_movies
    else:
        raise NotImplementedError

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


def run_squares(reps=1):
    p = 3

    for dataset_name in ["NOISY_MOVIES", "SHIFTED_MOVIES"]:

        print("#" * 40)
        print(dataset_name.center(40, "#"))
        print("#" * 40)

        arma_data, lstm_data = get_data(p, dataset_name)

        for rep in reversed(list(range(reps))):

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
    run_squares()
