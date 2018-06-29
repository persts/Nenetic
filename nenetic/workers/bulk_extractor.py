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
from PyQt5 import QtCore
from multiprocessing import Process, Queue, cpu_count


class BulkExtractor(QtCore.QThread):
	progress = QtCore.pyqtSignal(int)

	def __init__(self, extractor, image):
		QtCore.QThread.__init__(self)
		self.image = image
		self.extractor = extractor
		self.extractor.preprocess(self.image)
		self.vectors = [x for x in range(self.image.shape[0])]

		self.queue = Queue()
		self.processes = []
		self.threads = cpu_count() - 1
		for i in range(self.threads):
			rows = [x for x in range(i, self.image.shape[0], self.threads)]
			p = Process(target=self.extract, args=(rows, ))
			self.processes.append(p)

	def extract(self, rows):
		count = 0
		for row in rows:
			self.queue.put([row, self.extractor.extract_row(row)])
			count += 1
		#self.queue.put([None, None])
		return True

	def run(self):
		for p in self.processes:
			p.start()
		
		for p in self.processes:
			p.join()

		self.queue.put([None, None])
		'''
		progress = 0
		terminate = 0
		while terminate < self.threads:
			row, vector = self.queue.get()
			if row is None:
				terminate += 1
			else:
				progress += 1
				self.vectors[row] = vector
				self.progress.emit(progress)
		'''

	def buffer(self):
		for vector in self.vectors:
			yield vector
