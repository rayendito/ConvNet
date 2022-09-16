import os
from PIL import Image

def load_cats_and_dogs():
    image_data = {
        'label_names' : ['cat', 'dog'],
        'labels' : [],
        'data' : [],
    }

    classes = ["cat", "dog"]
    class_code = {
        'cat' : 0,
        'dog' : 1,
    }

    for kind in classes:
        for filename in os.scandir("data/{}".format(kind)):
            image_data['data'].append(load_one_image(filename.path))
            image_data['labels'].append(class_code[kind])
    return image_data

def load_one_image(path):
    im = Image.open(r"{}".format(path))
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    return pixels