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
from nenetic.extractors import Neighborhood


class Region(Neighborhood):
    def __init__(self, layer_definitions=[], pad=14, force_cpu=False):
        Neighborhood.__init__(self, layer_definitions=layer_definitions, pad=pad, force_cpu=force_cpu)

        self.type = 'raster'
        self.name = 'Region'

    def extract_at(self, x, y):
        X = x + (2 * self.pad) + 1
        Y = y + (2 * self.pad) + 1
        return self.stack[y:Y, x:X]
