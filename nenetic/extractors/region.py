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
# Nenetic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Nenetic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with with this software.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------
from nenetic.extractors import Neighborhood


class Region(Neighborhood):
    def __init__(self, layer_definitions=[], pad=14):
        Neighborhood.__init__(self, layer_definitions=layer_definitions, pad=pad)

        self.type = 'raster'
        self.name = 'Region'

    def extract_at(self, x, y, dst=None):
        X = x + (2 * self.pad) + 1
        Y = y + (2 * self.pad) + 1
        if dst is None:
            return self.stack[y:Y, x:X]
        else:
            dst[:] = self.stack[y:Y, x:X]
