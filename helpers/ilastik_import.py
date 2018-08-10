import os
import sys
import json
import numpy as np
from PIL import Image
from random import random

if len(sys.argv) < 3:
	print("Usage: ilastik_import ilastik_export_bmp original_image_name")
	sys.exit()

FILE = sys.argv[2]
img = Image.open(sys.argv[1])
img = np.array(img)
package = {'classes': [], 'points': {FILE: {}}, 'colors': {}}
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        value = img[y, x]
        if value > 0:
            value = str(value)
            if value not in package['classes']:
                package['classes'].append(value)
                package['colors'][value] = [int(random() * 255), int(random() * 255), int(random() * 255)]
                package['points'][FILE][value] = []
            package['points'][FILE][value].append({'x': x, 'y': y})
file = open(FILE + '.pnt', 'w')
json.dump(package, file)
file.close()
