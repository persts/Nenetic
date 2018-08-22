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
from multiprocessing import Process, Queue, cpu_count, Value
from nenetic.extractors import Region, Neighborhood


def extract(queue, queue_limit, extractor, rows):
    count = 0
    for row in rows:
        queue.put([row, extractor.extract_row(row)])
        count += 1
        if count % 2 == 0:
            while queue.qsize() > queue_limit.value:
                time.sleep(1)
            if queue.qsize() == 0:
                with queue_limit.get_lock():
                    queue_limit.value += 1
    return True


class ExtractorQueue(QtCore.QThread):

    def __init__(self, image, extractor_name, extractor_kwargs, number_of_cores=0):
        QtCore.QThread.__init__(self)
        self.image = image
        self.extractor_name = extractor_name
        self.extractor_kwargs = extractor_kwargs

        self.queue = Queue()
        self.processes = []

        if number_of_cores == 0:
            self.threads = cpu_count() - 1
        else:
            self.threads = number_of_cores

        if self.extractor_name == 'Neighborhood':
            self.extractor = Neighborhood(**extractor_kwargs)
        else:
            self.extractor = Region(**extractor_kwargs)
        self.extractor.preprocess(image)

    def run(self):
        queue_limit = Value('i', 5)
        for i in range(self.threads):
            rows = [x for x in range(i, self.image.shape[0], self.threads)]
            p = Process(target=extract, args=(self.queue, queue_limit, self.extractor, rows))
            self.processes.append(p)

        for p in self.processes:
            p.start()

        for p in self.processes:
            p.join()

        if self.queue is not None:
            self.queue.put([None, None])
