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
import numpy as np

from nenetic.extractors import Vector


class Neighborhood(Vector):
    def __init__(self, pad=1):
        Vector.__init__(self)
        self.pad = pad

        self.name = 'Neighborhood'
        self.kwargs = {'pad': pad}

    def extract_row(self, row):
        if len(self.stack) > 0:
            cols = self.stack[0].shape[1] - (self.pad * 2)
            vector = np.array([self.extract_value(0, row)])
            for i in range(1, cols):
                entry = np.array([self.extract_value(i, row)])
                vector = np.vstack((vector, entry))
            return vector

    def extract_value(self, x, y):
        vector = []
        X = x + (2 * self.pad) + 1
        Y = y + (2 * self.pad) + 1
        for image in self.stack:
            vector += image[y:Y, x:X].flatten().tolist()
        return vector

    def preprocess(self, image):
        self.stack = [np.pad(image, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric') / 255]
