import os
import sys
import json
import numpy as np
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: ilastick_export pnt_file_name")
    sys.exit()

DIRECTORY, FILE = os.path.split(sys.argv[1])

file = open(sys.argv[1], 'r')
data = json.load(file)
file.close()

points = None
if 'points' in data:
    points = data['points']
else:
    points = data['images']
for image in points:
    img = Image.open(os.path.join(DIRECTORY, image))
    img = np.array(img)
    img = img * 0.0
    for class_name in points[image]:
        c = [data['classes'].index(class_name) + 1] * 3
        for point in points[image][class_name]:
            img[int(point['y']), int(point['x'])] = c
    img = Image.fromarray(np.uint8(img[:, :, 0]))
    img.save(image + '_exported.bmp')
