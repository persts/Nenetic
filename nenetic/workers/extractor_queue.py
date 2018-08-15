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
from PyQt5 import QtCore
from multiprocessing import Process, Queue, cpu_count
from nenetic.extractors import *  # noqa: F403, F401

GPU = False
try:
    import cupy
    GPU = True
except ImportError:
    pass

def extract(queue, image, rows, extractor_name, extractor_kwargs):
    extractor = globals()[extractor_name](**extractor_kwargs)
    extractor.preprocess(image)
    count = 0
    for row in rows:
        queue.put([row, extractor.extract_row(row)])
        count += 1
        if count % 20 == 0:
            while queue.qsize() > 100:
                time.sleep(5)
    return True


class ExtractorQueue(QtCore.QThread):

    def __init__(self, image, extractor_name, extractor_kwargs, force_cpu=False):
        QtCore.QThread.__init__(self)
        self.image = image
        self.extractor_name = extractor_name
        self.extractor_kwargs = extractor_kwargs
        self.extractor_kwargs['force_cpu'] = force_cpu

        self.queue = Queue()
        self.processes = []
        if GPU and not force_cpu:
            self.threads = 2
        else:
            self.threads = cpu_count() - 1

    def run(self):
        for i in range(self.threads):
            rows = [x for x in range(i, self.image.shape[0], self.threads)]
            p = Process(target=extract, args=(self.queue, self.image, rows, self.extractor_name, self.extractor_kwargs))
            self.processes.append(p)

        for p in self.processes:
            p.start()

        for p in self.processes:
            p.join()

        if self.queue is not None:
            self.queue.put([None, None])
