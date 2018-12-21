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
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageQt
from PyQt5 import QtCore, QtGui

from nenetic.workers import ExtractorPool


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
            self.extractor_type = data['extractor']['type']
            self.extractor_name = data['extractor']['name']
            self.extractor_kwargs = data['extractor']['kwargs']

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
            self.feedback.emit('Classifier', 'Preparing extractors')
            extractor_pool = ExtractorPool(self.image, self.extractor_name, self.extractor_kwargs)
            extractor_pool.start()
            self.result = np.zeros((self.image.shape[0], self.image.shape[1], 3))
            while not extractor_pool.ready:
                self.sleep(1)  # Wait for the preprocessing to complete and for something to be on the queue
            processes = len(extractor_pool.processes)
            self.feedback.emit('Classifier', 'Classifying...')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            graph = tf.Graph()
            with graph.as_default():
                with tf.Session(config=config) as sess:
                    meta = tf.train.import_meta_graph(self.model)
                    meta.restore(sess, tf.train.latest_checkpoint(self.directory))

                    X = graph.get_tensor_by_name('Placeholder:0')
                    prediction = graph.get_tensor_by_name('prediction:0')
                    keep_prob = None
                    if self.extractor_type == 'raster':
                        keep_prob = graph.get_tensor_by_name('keep_prob:0')
                    progress = 0
                    while extractor_pool.isRunning():
                        count = 0
                        for c in range(processes):
                            row = extractor_pool.states[c].value
                            if row >= 0:
                                with extractor_pool.states[c].get_lock():
                                    count += 1
                                    if self.extractor_type == 'raster':
                                        predictions = sess.run(prediction, feed_dict={X: extractor_pool.arrays[c], keep_prob: 1.0})
                                    else:
                                        predictions = sess.run(prediction, feed_dict={X: extractor_pool.arrays[c]})
                                    extractor_pool.states[c].value = -1
                                for i in range(self.image.shape[1]):
                                    p = np.argmax(predictions[i])
                                    if predictions[i][p] >= self.threshold:
                                        class_name = self.classes[p]
                                        self.result[row, i] = self.colors[class_name]
                                progress += 1
                                self.progress.emit(progress)
                                if progress % 20 == 0:
                                    self.prep_update()
                            if self.stop:
                                for p in extractor_pool.processes:
                                    p.terminate()
                                while extractor_pool.isRunning():
                                    self.sleep(1)
                                break
                                self.feedback.emit('Classifier', 'Classification interrupted.')
                        if count == 0:
                            self.msleep(500)
                self.feedback.emit('Classifier', 'Classification completed.')
            self.prep_update()
            extractor_pool = None
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
