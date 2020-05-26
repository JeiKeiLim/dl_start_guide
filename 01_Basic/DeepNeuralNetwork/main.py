import tensorflow as tf
import numpy as np
from neural_network import ANN

import argparse

import pickle


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28)
x_train, x_test = x_train / 255.0, x_test / 255.0



def print_test_result(model_):
    print(np.argmax(model_.forward(x_train[0:10]), axis=1))
    print(np.argmax(y_train[0:10], axis=1))

    print("Train accuracy : {:.3f}%".format(
        (np.sum(np.argmax(model_.forward(x_train), axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]) * 100))
    print("Test accuracy : {:.3f}%".format(
        (np.sum(np.argmax(model_.forward(x_test), axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]) * 100))


def load_model(file_name):
    with open(file_name, "rb") as f:
        model_ = pickle.load(f)

    return model_


def training(model_, lr_decay=1.0, n_try=50, epoch_per_try=10):
    for i in range(n_try):
        batch_s = np.random.randint(1, 10000)
        lr = (0.5 / batch_s) * lr_decay

        print("Start training ... ({:02d}/{:02d}), Batch size: {:03d}, Learning rate : {:.5f}".format(i+1, n_try, batch_s, lr))
        model_.fit(x_train, y_train, epoch=epoch_per_try, learning_rate=lr, batch_size=batch_s)
        lr_decay = lr_decay * 0.99

        print_test_result(model_)

    with open("mnist_model.pkl", "wb") as f:
        pickle.dump(model_, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artificial Neural Network")
    parser.add_argument("--training", dest='training', action='store_true')
    parser.add_argument("--no-training", dest='training', action='store_false')
    parser.add_argument("--load_model", dest='load_model', action='store_true')
    parser.add_argument("--no-load_model", dest='load_model', action='store_false')
    parser.add_argument("--lr_decay", type=float, default=1.0)
    parser.add_argument("--n_try", type=int, default=50)
    parser.add_argument("--epoch_per_try", type=int, default=10)
    parser.set_defaults(training=True, load_model=False)
    args = parser.parse_args()

    if args.load_model:
        model = load_model("mnist_model.pkl")
    else:
        model = ANN(layer=[[(784, 257), 'relu'],
                           [(256, 129), 'relu'],
                           [(128, 10), 'sigmoid']
                           ])

    if args.training:
        training(model, lr_decay=args.lr_decay, n_try=args.n_try, epoch_per_try=args.epoch_per_try)
    else:
        print_test_result(model)

