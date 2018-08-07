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

from PyQt5 import QtCore
from random import shuffle
from PIL import Image


class Region(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    feedback = QtCore.pyqtSignal(str, str)

    def __init__(self, pad=15):
        QtCore.QObject.__init__(self)
        self.classes = []
        self.points = {}
        self.directory = ''
        self.data = []
        self.labels = []
        self.colors = {}

        self.pad = pad
        self.padded = None

        self.max_value = 255

        self.type = 'raster'
        self.name = 'Region'
        self.kwargs = {'pad': pad}

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
        self.points = packaged_points['points']
        self.colors = packaged_points['colors']

    def extract(self):
        self.data = []
        self.labels = []
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
                    vector = self.extract_region(int(point['x']), int(point['y']))
                    self.data.append(vector)
                    self.labels.append(label)
                    progress += 1
                    self.progress.emit(progress)

    def extract_row(self, row):
        if self.padded is not None:
            cols = self.padded.shape[1] - (self.pad * 2)
            vector = np.array([self.extract_region(0, row)])
            for i in range(1, cols):
                entry = np.array([self.extract_region(i, row)])
                vector = np.vstack((vector, entry))
            return vector

    def extract_region(self, x, y):
        X = x + (2 * self.pad) + 1
        Y = y + (2 * self.pad) + 1
        return self.padded[y:Y, x:X]

    def preprocess(self, image):
        self.padded = np.pad(image, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric')

    def save(self, file_name):
        self.feedback.emit('Extractor', 'Preparing to save data.')
        self.shuffle()
        data = []
        self.feedback.emit('Extractor', 'Converting image arrays to lists.')
        for entry in self.data:
            data.append(entry.tolist())
        package = {'classes': self.classes, 'labels': self.labels, 'data': data, 'colors': self.colors, 'extractor': {'name': self.name, 'type': self.type, 'kwargs': self.kwargs}}
        self.feedback.emit('Extractor', 'Writing to disk...')
        file = open(file_name, 'w')
        json.dump(package, file)
        file.close()
        self.feedback.emit('Extractor', 'Done.')

    def shuffle(self):
        shuffle_index = [x for x in range(len(self.data))]
        shuffle(shuffle_index)
        data = [0] * len(self.data)
        labels = [0] * len(self.data)
        for i in range(len(self.data)):
            data[i] = self.data[shuffle_index[i]]
            labels[i] = self.labels[shuffle_index[i]]
        self.data = data
        self.labels = labels
