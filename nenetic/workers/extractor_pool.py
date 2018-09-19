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
import time
import ctypes
import numpy as np
from PyQt5 import QtCore
from multiprocessing import Process, Value, Array
from nenetic.extractors import Region, Neighborhood

import psutil


def extract(extractor, rows, mem, state, shape):
    array = np.frombuffer(mem.get_obj(), dtype=np.float32).reshape(shape)
    for row in rows:
        with state.get_lock():
            extractor.extract_row(row, array)
            state.value = row
        while state.value != -1:
            time.sleep(0.5)
    with state.get_lock():
        state.value = -2
    return True


class ExtractorPool(QtCore.QThread):

    def __init__(self, image, extractor_name, extractor_kwargs):
        QtCore.QThread.__init__(self)
        self.image = image
        self.extractor_name = extractor_name
        self.extractor_kwargs = extractor_kwargs

        self.processes = []
        self.states = []
        self.mem = []
        self.arrays = []

        self.ready = False

    def run(self):
        if self.extractor_name == 'Neighborhood':
            self.extractor = Neighborhood(**self.extractor_kwargs)
        else:
            self.extractor = Region(**self.extractor_kwargs)
        self.extractor.preprocess(self.image)
        vector = self.extractor.extract_at(0, 0)
        shape = ((self.image.shape[1], ) + vector.shape)
        mem_size = 1
        for s in shape:
            mem_size *= s
        mem_available = psutil.virtual_memory().available / 1024 / 1024 / 2
        row_size = (mem_size * 4) / 1024 / 1024
        max_children = int(mem_available / row_size)
        if max_children == 0:
            max_children = 1
        elif max_children > 250:
            max_children = 250
        for i in range(max_children):
            state = Value('i', -1)
            mem = Array(ctypes.c_float, mem_size)
            array = np.frombuffer(mem.get_obj(), dtype=np.float32).reshape(shape)
            rows = [x for x in range(i, self.image.shape[0], max_children)]
            p = Process(target=extract, args=(self.extractor, rows, mem, state, shape))
            self.processes.append(p)
            self.mem.append(mem)
            self.states.append(state)
            self.arrays.append(array)

        self.ready = True

        for p in self.processes:
            p.start()

        for p in self.processes:
            p.join()
