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

from PyQt5 import QtCore
from tabulate import tabulate


class FcTrainer(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    feedback = QtCore.pyqtSignal(str, str)

    def __init__(self, data, directory, model_parameters):
        QtCore.QThread.__init__(self)
        self.stop = False
        self.classes = data['classes']
        self.colors = data['colors']
        self.extractor = data['extractor']
        self.epochs = model_parameters['epochs']
        self.learning_rate = model_parameters['learning_rate']
        self.l1_hidden_nodes = model_parameters['l1_hidden_nodes']
        self.l2_hidden_nodes = model_parameters['l2_hidden_nodes']
        self.batch_size = model_parameters['batch_size']

        self.directory = directory

        split = int(len(data['data']) * (1.0 - model_parameters['validation_split']))
        self.training_data = data['data'][0:split]
        self.training_labels = data['labels'][0:split]
        self.validation_data = data['data'][split:]
        self.validation_labels = data['labels'][split:]
        self.n_input = len(self.training_data[0])
        self.n_classes = len(self.training_labels[0])

    def confusion_matrix(self, predictions, labels):
        matrix = np.zeros((len(self.classes), len(self.classes)))
        for label, prediction in zip(labels, predictions):
            matrix[np.argmax(label), np.argmax(prediction)] += 1
        header = [''] + self.classes
        data = []
        for x in range(len(self.classes)):
            data.append([self.classes[x]] + matrix[x].tolist())
        return data, header

    def fc(self, x, length):
        W = tf.Variable(tf.truncated_normal([x.shape[1].value, length], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([length], stddev=0.1))
        return tf.matmul(x, W) + b

    def run(self):
        # Save the log file
        log = open(os.path.join(self.directory, 'log.txt'), 'w')
        log.write("Epochs: {}\n".format(self.epochs))
        log.write("Learning Rate: {}\n".format(self.learning_rate))
        log.write("L1 Hidden Nodes: {}\n".format(self.l1_hidden_nodes))
        log.write("L2 Hidden Nodes: {}\n".format(self.l2_hidden_nodes))
        log.write("Batch Size: {}\n".format(self.batch_size))
        log.write("\n")
        X = tf.placeholder(tf.float32, [None, self.n_input])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

        layer_1 = tf.nn.relu(self.fc(X, self.l1_hidden_nodes))
        layer_2 = tf.nn.relu(self.fc(layer_1, self.l2_hidden_nodes))
        layer_out = self.fc(layer_2, self.n_classes)

        logits = layer_out
        prediction = tf.nn.softmax(logits, name='prediction')

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            total_batch = int(len(self.training_data) / self.batch_size) + 1
            batch_x = []
            batch_y = []
            for i in range(total_batch):
                batch_x.append(np.array(self.training_data[self.batch_size * i:self.batch_size * (i + 1)]))
                batch_y.append(np.array(self.training_labels[self.batch_size * i:self.batch_size * (i + 1)]))
            for epoch in range(self.epochs):
                avg_loss = 0
                avg_acc = 0
                for i in range(total_batch):
                    sess.run([train_op], feed_dict={X: batch_x[i], Y: batch_y[i]})
                    if epoch % 100 == 0:
                        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x[i], Y: batch_y[i]})
                        avg_loss += loss / total_batch
                        avg_acc += acc / total_batch
                if epoch % 100 == 0:
                    message = 'Epoch: {} Avg Batch [loss: {:.4f}  acc: {:.3f}]'.format(epoch, avg_loss, avg_acc)
                    self.feedback.emit('Train', message)
                    log.write(message + "\n")
                self.progress.emit(epoch + 1)
                if self.stop:
                    self.feedback.emit('Train', 'Training interrupted.')
                    log.write('Training interrupted.')
                    break

            pred_train = sess.run(prediction, feed_dict={X: self.training_data})
            pred_validation = sess.run(prediction, feed_dict={X: self.validation_data})

            message = "Train Acc: {:.5f}".format(sess.run(accuracy, feed_dict={X: self.training_data, Y: self.training_labels}))
            self.feedback.emit('Train', message)
            log.write("\n" + message + "\n")

            data, header = self.confusion_matrix(pred_train, self.training_labels)
            output = tabulate(data, headers=header, tablefmt='orgtbl')
            log.write("Training data confusion matrix\n")
            log.write(output + "\n\n")

            message = "Validation Acc: {:.5f}".format(sess.run(accuracy, feed_dict={X: self.validation_data, Y: self.validation_labels}))
            self.feedback.emit('Train', message)
            log.write(message + "\n")

            data, header = self.confusion_matrix(pred_validation, self.validation_labels)
            output = tabulate(data, headers=header, tablefmt='orgtbl')
            log.write("Training data confusion matrix\n")
            log.write(output + "\n")
            self.feedback.emit('Train', 'See log for confusion matrix')

            # Save the Model
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(self.directory, 'model'))

            # Save the metadata
            file = open(os.path.join(self.directory, 'nenetic-metadata.json'), 'w')
            package = {'classes': self.classes, 'colors': self.colors, 'extractor': self.extractor}
            json.dump(package, file)
            file.close()

            # Close log file
            log.close()
