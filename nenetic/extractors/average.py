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
from scipy.signal import convolve2d as c2d

GPU = False
try:
    import cupy
    GPU = True
except ImportError:
    pass


class Average(Vector):
    def __init__(self, kernels=5, solid_kernel=True, force_cpu=False):
        Vector.__init__(self, force_cpu=force_cpu)
        self.kernels = []

        self.name = 'Average'
        self.kwargs = {'kernels': kernels, 'solid_kernel': solid_kernel}

        for size in range(3, (kernels * 2) + 3, 2):
            kernel = np.ones((size, size))
            if not solid_kernel:
                kernel[1:size - 1, 1:size - 1] = 0
            kernel = kernel / np.sum(kernel)
            self.kernels.append(kernel)

    def preprocess(self, image):
        self.stack = image / self.max_value
        for kernel in self.kernels:
            bands = []
            for band in range(image.shape[2]):
                b = c2d(image[:, :, 0], kernel, mode='same')
                bands.append(b)
            img = np.dstack(bands)
            img = img / self.max_value
            self.stack = np.dstack((self.stack, img))
        self.stack = self.stack.astype(np.float32)
        if GPU and not self.force_cpu:
            self.stack = cupy.array(self.stack)
