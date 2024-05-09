

"""This file using libraries"""
import numpy as np
from keras._tf_keras.keras.models import model_from_json
from keras._tf_keras.keras.preprocessing import image


def get_prediction(path):

    file = open("model_file.json", "r")

    read_file = file.read()

    file.close()

    model = model_from_json(read_file)

    model.load_weights("model.weights.h5")

    label = ['1', ')', 'd', 'geq', 'X', 'gamma', 'S', 'lambda', 'y', 'A', ']', 'ascii_124', 'phi',
             'v', 'C', '=', 'b', 'q', 'log', '}', 'times', 'int', 'gt', 'beta', ',', 'lt', '0',
             '2', 'cos', 'k', 'e', '-', 'forall', 'pm', '[', 'G', 'u', '!', '3', 'f', '4', 'p',
             'j', 'alpha', '+', 'Delta', 'exists', '9', '8', 'sin', 'o', 'R', '(', 'z', 'leq', 'T',
             'M', 'neq', 'sqrt', 'pi', 'prime', 'mu', 'infty', 'H', 'sum', 'l', 'rightarrow', 'sigma',
             'in', 'ldots', '6', 'N', 'lim', '7', 'forward_slash', '5', 'tan', 'i', 'theta',
             'div', '{', 'w']

    test_image = image.load_img(path, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)
    classes = label[prediction.argmax()]
    return classes
