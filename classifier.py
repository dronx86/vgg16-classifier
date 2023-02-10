import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from PIL import Image
import json
import subprocess

with open("classes.txt", "r") as file:
    classes_str = file.read()

CLASSES = json.loads(classes_str)
SIZE = (224, 224)


def img_normalization(img_file):
    with Image.open(str(img_file)) as img:
        width, height = img.size
        delta = abs(width - height)/2

        if width > height:
            box = (delta, 0, width-delta, height)
        else:
            box = (0, delta, width, height-delta)

        norm_img = img.resize(SIZE, box=box)
        return norm_img


def classify(norm_img):
    image = np.array(norm_img)
    input_value = keras.applications.vgg16.preprocess_input(image)
    input_value = np.expand_dims(input_value, axis=0)
    model = keras.applications.VGG16()
    res = model.predict(input_value)
    index = str(np.argmax(res)+1)
    res_name = CLASSES[index]
    return res_name


def main():
    import sys
    image_file = sys.argv[1]
    norm_image = img_normalization(image_file)
    res = classify(norm_image)
    print(res)
    norm_image.show()


if __name__ == "__main__":
    main()
