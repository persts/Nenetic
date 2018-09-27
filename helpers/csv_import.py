import csv
import sys
import json
from PIL import Image
from random import randint

if len(sys.argv) < 3:
    print("Usage: csv_import csv_file original_image_name     **CSV format expected to be X, Y, Class")
    sys.exit()

FILE = sys.argv[2]
img = Image.open(FILE)
height = float(img.size[1])
package = {'classes': [], 'points': {FILE: {}}, 'colors': {}}
with open(sys.argv[1]) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)
    for row in reader:
        value = 'class_' + row[2]
        if value not in package['classes']:
            package['classes'].append(value)
            package['colors'][value] = [randint(0, 255), randint(0, 255), randint(0, 255)]
            package['points'][FILE][value] = []
        package['points'][FILE][value].append({'x': float(row[0]), 'y': height - float(row[1])})
package['classes'].sort()
file = open(FILE + '.pnt', 'w')
json.dump(package, file)
file.close()
