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


class Trainer(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    feedback = QtCore.pyqtSignal(str, str)

    def __init__(self, data, directory, model_parameters):
        QtCore.QThread.__init__(self)
        self.classes = data['classes']
        self.colors = data['colors']
        self.extractor = data['extractor']
        self.epochs = model_parameters['epochs']
        self.learning_rate = model_parameters['learning_rate']
        self.l1_hidden_nodes = model_parameters['l1_hidden_nodes']
        self.l2_hidden_nodes = model_parameters['l2_hidden_nodes']
        self.batch_size = 256

        self.directory = directory

        split = int(len(data['data']) * ( 1.0 - model_parameters['validation_split']))
        self.training_data = data['data'][0:split]
        self.training_labels = data['labels'][0:split]
        self.validation_data = data['data'][split:]
        self.validation_labels = data['labels'][split:]
        self.n_input = len(self.training_data[0])
        self.n_classes = len(self.training_labels[0])

    def run(self):
        self.log = "Epochs: {}\n".format(self.epochs)
        self.log += "Learning Rate: {}\n".format(self.learning_rate)
        self.log += "L1 Hidden Nodes: {}\n".format(self.l1_hidden_nodes)
        self.log += "L2 Hidden Nodes: {}\n".format(self.l2_hidden_nodes)
        self.log += "Batch Size: {}\n".format(self.batch_size)
        self.log += "\n"
        X = tf.placeholder(tf.float32, [None, self.n_input])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

        W1 = tf.Variable(tf.random_normal([self.n_input, self.l1_hidden_nodes], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random_normal([self.l1_hidden_nodes]), name='b1')

        W2 = tf.Variable(tf.random_normal([self.l1_hidden_nodes, self.l2_hidden_nodes], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random_normal([self.l2_hidden_nodes]), name='b2')

        W_out = tf.Variable(tf.random_normal([self.l2_hidden_nodes, self.n_classes], stddev=0.03), name='W_out')
        b_out = tf.Variable(tf.random_normal([self.n_classes]), name='b_out')

        layer_1 = tf.add(tf.matmul(X, W1), b1)
        layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
        layer_out = tf.matmul(layer_2, W_out) + b_out

        logits = layer_out
        prediction = tf.nn.softmax(logits, name='prediction')

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
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
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x[i], Y: batch_y[i]})
                    if epoch % 50 == 0:
                        avg_loss += loss / total_batch
                        avg_acc += acc / total_batch
                if epoch % 50 == 0:
                    message = 'Epoch: {} batch loss: {:.5f} batch acc: {:.3f}'.format(epoch, avg_loss, avg_acc)
                    self.feedback.emit('Train', message)
                    self.log += message + "\n"
                self.progress.emit(epoch + 1)

            message = "Train Acc: {:.5f}".format(sess.run(accuracy, feed_dict={X: self.training_data, Y: self.training_labels}))
            self.feedback.emit('Train', message)
            self.log += "\n" + message + "\n"
            message = "Validation Acc: {:.5f}".format(sess.run(accuracy, feed_dict={X: self.validation_data, Y: self.validation_labels}))
            self.feedback.emit('Train', message)
            self.log += message + "\n"

            # Save the Model
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(self.directory, 'model'))

            # Save the metadata
            file = open(os.path.join(self.directory, 'nenetic-metadata.json'), 'w')
            package = {'classes': self.classes, 'colors': self.colors, 'extractor': self.extractor}
            json.dump(package, file)
            file.close()

            # Save the log file
            file = open(os.path.join(self.directory, 'log.txt'), 'w')
            file.write(self.log)
            file.close()
