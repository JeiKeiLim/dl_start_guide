import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from tfhelper.gpu import allow_gpu_memory_growth
from tfhelper.tflite import predict_tflite_interpreter, evaluate_tflite_interpreter
from tfhelper.tflite import load_pruned_model
# from .a01_prune_model import get_model_cifar
import pathlib
import time


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def time_measure(f, *args, verbose=0, prefix=""):
    start_time = time.time()
    result = f(*args)
    end_time = time.time()

    if verbose > 0:
        print("{} Took {:.3f}s".format(prefix, end_time-start_time))

    return result, end_time-start_time


def quantize_from_existing_model():
    # Refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quant.ipynb

    model = tf.keras.models.load_model("./export/saved_model/base_model.h5")
    model = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(32, 32, 3)), model])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # loss, base_accuracy = model.evaluate(x_test, y_test)
    result, time = time_measure(model.evaluate, x_test, y_test, verbose=1, prefix="Base model ")
    loss, base_accuracy = result

    model = load_pruned_model("./export/saved_model/pruned_model.h5")
    model = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(32, 32, 3)), model])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_models_dir = pathlib.Path("./export/saved_model/converted_model/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    tflite_file = tflite_models_dir / "converted_model.tflite"
    tflite_file.write_bytes(converter.convert()) # Writing tflite model

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_file = tflite_models_dir / "converted_model_quantized.tflite"
    tflite_quantized_file.write_bytes(converter.convert())

    tflite_interpreter = tf.lite.Interpreter(model_path="./export/saved_model/converted_model/converted_model.tflite")
    tflite_interpreter.allocate_tensors()

    # evaluate_tflite_interpreter(tflite_interpreter, x_test, y_test)
    result, time = time_measure(evaluate_tflite_interpreter, tflite_interpreter, x_test, y_test, verbose=1, prefix="Non quantized ")
    tflite_accuracy, predictions = result

    quantized_tflite_interpreter = tf.lite.Interpreter(
        model_path="./export/saved_model/converted_model/converted_model_quantized.tflite")
    quantized_tflite_interpreter.allocate_tensors()

    # evaluate_tflite_interpreter(tflite_interpreter, x_test, y_test)
    result, time = time_measure(evaluate_tflite_interpreter, quantized_tflite_interpreter, x_test, y_test, verbose=1, prefix="Quantized ")
    quantized_accuracy, predictions = result

    print("Base Model Accuracy: {:.3f}%\nPruned TFLite Model Accuracy: {:.3f}%\nQuantized TFLite Model Accuracy: {:.3f}%".format(base_accuracy*100, quantized_accuracy*100, quantized_accuracy*100))


if __name__ == '__main__':
    quantize_from_existing_model()
    # model = get_model_cifar()
