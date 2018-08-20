# -*- coding: utf-8 -*-
#
# Neural Network Image Classifier (Nenetic)
# Copyright (C) 2018 Peter Ersts
# ersts@amnh.org
#
# --------------------------------------------------------------------------
#
# This file is part of Neural Network Image Classifier (Nenetic).
#
# Andenet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Andenet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with with this software.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------
import os
import json
import numpy as np
import pickle

from PyQt5 import QtCore
from random import shuffle
from PIL import Image

from nenetic.extractors import Generator

GPU = False
try:
    import cupy
    GPU = True
except ImportError:
    pass


class Vector(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    feedback = QtCore.pyqtSignal(str, str)

    def __init__(self, layer_definitions=[], pad=0, force_cpu=False):
        QtCore.QObject.__init__(self)
        self.classes = []
        self.points = {}
        self.directory = ''
        self.data = []
        self.labels = []
        self.colors = {}

        self.layer_definitions = layer_definitions
        self.pad = pad
        self.force_cpu = force_cpu

        self.stack = None
        self.generator = Generator()

        self.type = 'vector'
        self.name = 'Vector'
        self.kwargs = {'layer_definitions': layer_definitions, 'pad': pad}

    def load(self, file_name):
        self.directory = os.path.split(file_name)[0]
        file = open(file_name, 'r')
        points = json.load(file)
        file.close()
        self.load_points(points)

    def load_points(self, packaged_points, directory=None):
        if directory is not None:
            self.directory = directory
        self.classes = packaged_points['classes']
        if 'images' in packaged_points:
            # Backward compatibility
            self.points = packaged_points['images']
        else:
            self.points = packaged_points['points']
        self.colors = packaged_points['colors']

    def extract(self):
        self.data = np.array(None)
        self.labels = []
        shape = ()
        progress = 0
        for image in self.points:
            try:
                img = Image.open(os.path.join(self.directory, image))
            except OSError as e:
                self.feedback.emit('Extractor', image + ' could not be opened, skipping...')
                break
            array = np.array(img)
            img.close()
            self.feedback.emit('Extractor', 'Preprocessing image -> {}'.format(image))
            self.preprocess(array)
            self.feedback.emit('Extractor', 'Extacting points')
            for class_name in self.points[image]:
                label = [0] * len(self.classes)
                label[self.classes.index(class_name)] = 1
                points = self.points[image][class_name]
                for point in points:
                    vector = self.extract_at(int(point['x']), int(point['y']))
                    if self.data.shape == ():
                        if GPU and not self.force_cpu:
                            self.data = cupy.reshape(vector, (1,) + vector.shape)
                        else:
                            self.data = np.reshape(vector, (1,) + vector.shape)
                        shape = self.data.shape
                    else:
                        if GPU and not self.force_cpu:
                            self.data = cupy.vstack((self.data, cupy.reshape(vector, shape)))
                        else:
                            self.data = np.vstack((self.data, np.reshape(vector, shape)))
                    self.labels.append(label)
                    progress += 1
                    self.progress.emit(progress)
        if GPU and not self.force_cpu:
            self.data = cupy.asnumpy(self.data)

    def extract_row(self, row):
        if self.stack is not None:
            cols = self.stack.shape[1]
            vector = self.extract_at(0, row)
            vector = vector.reshape((1, ) + vector.shape)
            shape = vector.shape
            for i in range(1, cols):
                entry = self.extract_at(i, row).reshape(shape)
                if GPU and not self.force_cpu:
                    vector = cupy.vstack((vector, entry))
                else:
                    vector = np.vstack((vector, entry))
            if GPU and not self.force_cpu:
                vector = cupy.asnumpy(vector)
            return vector

    def extract_at(self, x, y):
        return self.stack[y, x]

    def preprocess(self, image):
        stack, _ = self.generator.generate(image, self.layer_definitions, self.pad)
        if GPU and not self.force_cpu:
            self.stack = cupy.array(stack.astype(np.float32))
        else:
            self.stack = stack.astype(np.float32)

    def save(self, file_name):
        self.feedback.emit('Extractor', 'Preparing to save data.')
        json_format = False
        if file_name[-4:].lower() == 'json':
            json_format = True
        self.shuffle()
        self.feedback.emit('Extractor', 'Writing to disk...')
        package = None
        if json_format:
            data = []
            for entry in self.data:
                data.append(entry.tolist())
            package = {'classes': self.classes, 'labels': self.labels, 'data': data, 'colors': self.colors, 'extractor': {'name': self.name, 'type': self.type, 'kwargs': self.kwargs}}
            file = open(file_name, 'w')
            json.dump(package, file)
            file.close()
        else:
            package = {'classes': self.classes, 'labels': self.labels, 'data': self.data, 'colors': self.colors, 'extractor': {'name': self.name, 'type': self.type, 'kwargs': self.kwargs}}
            file = open(file_name, 'wb')
            pickle.dump(package, file)
            file.close()
        self.feedback.emit('Extractor', 'Done.')

    def shuffle(self):
        shuffle_index = [x for x in range(self.data.shape[0])]
        shuffle(shuffle_index)
        data = np.zeros(self.data.shape)
        labels = [0] * self.data.shape[0]
        for i in range(self.data.shape[0]):
            data[i] = self.data[shuffle_index[i]]
            labels[i] = self.labels[shuffle_index[i]]
        self.data = data
        self.labels = labels
