import tensorflow as tf
import numpy as np
import tempfile
import tensorflow_model_optimization as tfmot
from tfhelper.gpu import allow_gpu_memory_growth
from tfhelper.tensorboard import run_tensorboard, wait_ctrl_c, get_tf_callbacks
from tfhelper.transfer_learning import get_transfer_learning_model
from tfhelper.tflite import load_pruned_model
import time


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def compile_model(model_, optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=('accuracy')):
    model_.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model_


def get_model():
    model_ = tf.keras.Sequential([
        tf.keras.layers.Reshape(target_shape=(28, 28, 3), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model_.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                   metrics=['accuracy'])

    return model_


def get_model_cifar():
    model_ = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME',  activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME',  activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model_.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                   metrics=['accuracy'])

    return model_


def train_model(model_, epochs=10, batch_size=128, save_file=True):

    model_.fit(x_train, y_train, epochs=epochs, validation_split=0.1, batch_size=batch_size)

    _, baseline_model_accuracy = model_.evaluate(x_test, y_test, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)

    if save_file:
        _, keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model_, keras_file, include_optimizer=False)
        print('Saved baseline model to:', keras_file)
    else:
        keras_file = None

    return baseline_model_accuracy, keras_file


def compute_sparsity(model_, sparse_threshold=0.05):
    sparsities = np.zeros(len(model_.layers))

    for i in range(sparsities.shape[0]):
        if len(model.layers[i].weights) < 1:
            sparsities[i] = np.nan
            continue

        sparse_index = np.argwhere(np.logical_and(model.layers[i].weights[0].numpy().flatten() < sparse_threshold,
                                                model.layers[i].weights[0].numpy().flatten() > -sparse_threshold))

        sparsities[i] = sparse_index.shape[0] / np.prod(model.layers[i].weights[0].shape)

    return sparsities


def prune_model(model_, epochs=5, batch_size=128, logdir=None, run_tensor_board=True):

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    validation_split = 0.1 # 10% of training set will be used for validation set.

    num_images = x_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                   final_sparsity=0.80,
                                                                   begin_step=0,
                                                                   end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model_, **pruning_params)

    model_for_pruning.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model_for_pruning.summary()

    if logdir is None:
        logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    if run_tensor_board:
        run_tensorboard(logdir)

    model_for_pruning.fit(x_train, y_train,
                          batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                          callbacks=callbacks)

    _, model_for_pruning_accuracy = model_for_pruning.evaluate(x_test, y_test, verbose=0)

    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_pruning, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)

    return model_for_pruning, model_for_pruning_accuracy, pruned_keras_file


def convert_to_tflite_model(pruned_model):
    model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()

    _, pruned_tflite_file = tempfile.mkstemp('.tflite')

    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)

    print('Saved pruned TFLite model to:', pruned_tflite_file)

    return model_for_export, pruned_tflite_file


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    print("File size before compression: {:.2f} bytes".format(os.path.getsize(file)))
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)


def get_model_evaluation_time(model_, x_test_, y_test_):
    start_time = time.time()
    loss, accuracy = model_.evaluate(x_test_, y_test_)
    end_time = time.time()

    eval_time = end_time-start_time

    print("Model Evaluation Time: {:.3f} seconds".format(eval_time))
    return loss, accuracy, eval_time


def transfer_learning_model():
    model_, base_model = get_transfer_learning_model(tf.keras.applications.VGG16)

    train_model(model_, epochs=1, batch_size=8, save_file=False)

    for i in range(len(base_model.layers) - 1, -1, -1):
        if type(base_model.layers[i]) != tf.keras.layers.Conv2D:
            continue

        base_model.layers[i].trainable = True

        unfrozen_list = []
        for j in range(len(base_model.layers)):
            if base_model.layers[j].trainable:
                unfrozen_list.append(j)

        print("Unfreezing layer {} (Shape: {}".format(i, base_model.layers[i].weights[0].shape))
        print("Currently unfrozen layer list: {}".format(unfrozen_list))

        tf.keras.backend.clear_session()

        model_ = compile_model(model_)
        train_model(model_, epochs=1, batch_size=8, save_file=False)
        if i == 16:
            break

    return model_


def train_from_scratch():
    allow_gpu_memory_growth()
    # model = get_model()
    # model = transfer_learning()
    model = get_model_cifar()
    model.summary()

    callbacks, log_root = get_tf_callbacks("./export/", confuse_callback=False, modelsaver_callback=False)

    run_tensorboard(log_root)
    model_accuracy, keras_file = train_model(model, epochs=10, batch_size=128, save_file=True)

    pruned_model, pruned_model_accuracy, pruned_keras_file = prune_model(model, batch_size=32,
                                                                         logdir="{}/pruning".format(log_root),
                                                                         run_tensor_board=False)
    pruned_tflite_model, pruned_tflite_file = convert_to_tflite_model(pruned_model)

    print('Baseline test accuracy:', model_accuracy)
    print('Pruned test accuracy:', pruned_model_accuracy)

    print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
    print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))

    compile_model(pruned_tflite_model)
    get_model_evaluation_time(model, x_test, y_test)
    get_model_evaluation_time(pruned_model, x_test, y_test)
    get_model_evaluation_time(pruned_tflite_model, x_test, y_test)

    wait_ctrl_c()


if __name__ == '__main__':
    train_from_scratch()
    # model = tf.keras.models.load_model("./export/saved_model/base_model.h5")
    # pruned_model = load_pruned_model("./export/saved_model/pruned_model.h5")

