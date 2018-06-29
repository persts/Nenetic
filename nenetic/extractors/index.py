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


class Index(Vector):
    def __init__(self, kernels=5, solid_kernel=True):
        Vector.__init__(self)

        self.name = 'Index'
        self.kwargs = {}

    def preprocess(self, image):
        #np.seterr(divide='ignore', invalid='ignore')
        self.stack = [image / 255]
        img = np.int32(image)
        bands = np.split(img, img.shape[2], axis=2)
        denom = np.clip(bands[1] + bands[0], 1, None)
        vndvi = (((bands[1] - bands[0]) / denom) + 1 ) / 2
        self.stack.append(vndvi)
        denom = np.clip(2 * bands[1] + bands[0] + bands[2], 1, None)
        gli = (((2 * bands[1] - bands[0] - bands[2]) / denom) + 1) / 2
        self.stack.append(gli)
        denom = np.clip(bands[1] + bands[0] - bands[2], 1, None)
        vari = (((bands[1] - bands[0]) / denom) + 1) / 2
        self.stack.append(vari)