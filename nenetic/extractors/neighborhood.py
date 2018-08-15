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

GPU = False
try:
    import cupy
    GPU = True
except ImportError:
    pass


class Neighborhood(Vector):
    def __init__(self, pad=0, force_cpu=False):
        Vector.__init__(self, force_cpu=force_cpu)
        self.pad = pad

        self.name = 'Neighborhood'
        self.kwargs = {'pad': pad}

    def extract_row(self, row):
        if self.stack is not None:
            cols = self.stack.shape[1] - (self.pad * 2)
            vector = self.extract_region(0, row)
            vector = vector.reshape((1, ) + vector.shape)
            shape = vector.shape
            for i in range(1, cols):
                entry = self.extract_region(i, row).reshape(shape)
                if GPU and not self.force_cpu:
                    vector = cupy.vstack((vector, entry))
                else:
                    vector = np.vstack((vector, entry))
            if GPU and not self.force_cpu:
                vector = cupy.asnumpy(vector)
            return vector

    def extract_value(self, x, y):
        return self.extract_region(x, y)

    def extract_region(self, x, y):
        X = x + (2 * self.pad) + 1
        Y = y + (2 * self.pad) + 1
        return self.stack[y:Y, x:X].flatten()

    def preprocess(self, image):
        self.stack = (np.pad(image, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='symmetric') / self.max_value).astype(np.float32)
        if GPU and not self.force_cpu:
            self.stack = cupy.array(self.stack)
