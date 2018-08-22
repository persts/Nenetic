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
    def __init__(self, layer_definitions=[], pad=0):
        Vector.__init__(self, layer_definitions=layer_definitions, pad=pad)
        self.name = 'Neighborhood'

    def extract_row(self, row):
        if self.stack is not None:
            vector = self.extract_at(0, row)
            cols = self.stack.shape[1] - (self.pad * 2)
            vector = np.ndarray((cols, ) + vector.shape)
            for i in range(cols):
                vector[i] = self.extract_at(i, row)
            return vector

    def extract_at(self, x, y):
        X = x + (2 * self.pad) + 1
        Y = y + (2 * self.pad) + 1
        return self.stack[y:Y, x:X].flatten()
