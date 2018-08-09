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
import json
import numpy as np
from nenetic.extractors import Neighborhood


class Region(Neighborhood):
    def __init__(self, pad=25, include_indices=False):
        Neighborhood.__init__(self)

        self.pad = pad
        self.include_indices = include_indices

        self.type = 'raster'
        self.name = 'Region'
        self.kwargs = {'pad': pad, 'include_indices': include_indices}

    def extract_region(self, x, y):
        X = x + (2 * self.pad) + 1
        Y = y + (2 * self.pad) + 1
        return self.stack[y:Y, x:X]

    def preprocess(self, image):
        stack = np.pad(image, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric') / self.max_value
        if self.include_indices:
            img = np.int32(image)
            bands = np.split(img, img.shape[2], axis=2)
            denom = np.clip(bands[1] + bands[0], 1, None)
            vndvi = (((bands[1] - bands[0]) / denom) + 1) / 2
            new_layer = np.pad(vndvi, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric')
            stack = np.dstack((stack, new_layer))

            denom = np.clip(2 * bands[1] + bands[0] + bands[2], 1, None)
            gli = (((2 * bands[1] - bands[0] - bands[2]) / denom) + 1) / 2
            new_layer = np.pad(gli, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric')
            stack = np.dstack((stack, new_layer))

            denom = np.clip(bands[1] + bands[0] - bands[2], 1, None)
            vari = (((bands[1] - bands[0]) / denom) + 1) / 2
            new_layer = np.pad(vari, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric')
            stack = np.dstack((stack, new_layer))

            average = ((bands[0] + bands[1] + bands[2]) / 3) / self.max_value
            new_layer = np.pad(average, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric')
            stack = np.dstack((stack, new_layer))

            luminosity = (0.21 * bands[0] + 0.72 * bands[1] + 0.07 * bands[2]) / self.max_value
            new_layer = np.pad(luminosity, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric')
            stack = np.dstack((stack, new_layer))

            maximum = np.maximum(bands[0], bands[1])
            maximum = np.maximum(maximum, bands[2])
            minimum = np.minimum(bands[0], bands[1])
            minimum = np.minimum(minimum, bands[2])
            lightness = ((maximum + minimum) / 2) / self.max_value
            new_layer = np.pad(lightness, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric')
            stack = np.dstack((stack, new_layer))

        self.stack = stack

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
