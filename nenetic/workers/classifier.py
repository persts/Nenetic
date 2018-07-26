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
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageQt
from PyQt5 import QtCore, QtGui

from nenetic.extractors import *
from nenetic.workers import BulkExtractor


class Classifier(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    feedback = QtCore.pyqtSignal(str, str)
    update = QtCore.pyqtSignal(QtGui.QPixmap)

    def __init__(self, model=None):
        QtCore.QThread.__init__(self)
        self.stop = False
        self.image = None
        self.result = None
        self.threshold = 0.9
        self.model = model

        if model is not None:
            self.directory = os.path.split(model)[0]
            # Need to test if file exists
            file = open(os.path.join(self.directory, "nenetic-metadata.json"))
            data = json.load(file)
            file.close()

            self.classes = data['classes']
            self.colors = {}
            for color in data['colors']:
                self.colors[color] = np.array(data['colors'][color])

            name = data['extractor']['name']
            kwargs = data['extractor']['kwargs']
            self.extractor = globals()[name](**kwargs)

    def load_classified_image(self, file_name):
        img = Image.open(file_name)
        self.result = np.array(img)
        self.prep_update()

    def prep_update(self):
        img = Image.fromarray(np.uint8(self.result))
        self.temp = ImageQt.ImageQt(img)
        self.update.emit(QtGui.QPixmap.fromImage(self.temp))

    def run(self):
        if self.model is not None:
            print('here')
            self.feedback.emit('Classifier', 'Preparing extractor')
            bulk_extractor = BulkExtractor(self.extractor, self.image)
            bulk_extractor.progress.connect(self.relay_progress)
            bulk_extractor.start()
            # self.feedback.emit('Classifier', 'Bulk extracting vectors')
            # while(bulk_extractor.isRunning()):
            #    self.sleep(1)
            self.result = np.zeros((self.image.shape[0], self.image.shape[1], 3))
            self.feedback.emit('Classifier', 'Classifying...')
            graph = tf.Graph()
            with graph.as_default():
                with tf.Session() as sess:
                    meta = tf.train.import_meta_graph(self.model)
                    meta.restore(sess, tf.train.latest_checkpoint(self.directory))

                    X = graph.get_tensor_by_name('Placeholder:0')
                    prediction = graph.get_tensor_by_name('prediction:0')

                    progress = 0
                    # for vector in bulk_extractor.buffer():
                    row, vector = bulk_extractor.queue.get()
                    while row is not None:
                        predictions = sess.run(prediction, feed_dict={X: vector})
                        for i in range(self.image.shape[1]):
                            p = np.argmax(predictions[i])
                            if predictions[i][p] >= self.threshold:
                                class_name = self.classes[p]
                                self.result[row, i] = self.colors[class_name]
                        progress += 1
                        self.progress.emit(progress)
                        if row % 50 == 0:
                            self.prep_update()
                        row, vector = bulk_extractor.queue.get()
                        if self.stop:
                            for p in bulk_extractor.processes:
                                p.terminate()
                            self.feedback.emit('Classifier', 'Classification interrupted.')
                            bulk_extractor.queue = None
                            return
                self.feedback.emit('Classifier', 'Classification completed.')
            self.prep_update()
        else:
            self.feedback.emit('Classifier', 'No model is loaded.')

    def relay_progress(self, value):
        self.progress.emit(value)

    def save_classification(self, file_name):
        if self.result is not None:
            img = Image.fromarray(np.uint8(self.result))
            img.save(file_name)
            self.feedback.emit('Classifier', 'Classifiation saved to -> {}'.format(file_name))
        else:
            self.feedback.emit('Classifier', 'Classification no yet available.')
