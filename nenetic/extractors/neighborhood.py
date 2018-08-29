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

    def extract_row(self, row, dst=None):
        if self.stack is not None:
            cols = self.stack.shape[1] - (self.pad * 2)
            if dst is None:
                vector = self.extract_at(0, row)
                vector = np.ndarray((cols, ) + vector.shape)
            else:
                vector = dst
            for i in range(cols):
                self.extract_at(i, row, dst=vector[i])
            return vector

    def extract_at(self, x, y, dst=None):
        X = x + (2 * self.pad) + 1
        Y = y + (2 * self.pad) + 1
        if dst is None:
            return self.stack[y:Y, x:X].flatten()
        else:
            dst[:] = self.stack[y:Y, x:X].flatten()
