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


class ConvTrainer(QtCore.QThread):
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
        self.final_layer_size = model_parameters['fc_size']
        self.model_definition = model_parameters['model']
        self.batch_size = 64

        self.directory = directory

        split = int(len(data['data']) * (1.0 - model_parameters['validation_split']))
        self.training_data = data['data'][0:split]
        self.training_labels = data['labels'][0:split]
        self.validation_data = data['data'][split:]
        self.validation_labels = data['labels'][split:]
        self.n_input = len(self.training_data[0])
        self.n_classes = len(self.training_labels[0])
        array = np.array(self.training_data[0])
        self.input_shape = array.shape

    def confusion_matrix(self, predictions, labels):
        matrix = np.zeros((len(self.classes), len(self.classes)))
        for label, prediction in zip(labels, predictions):
            matrix[np.argmax(label), np.argmax(prediction)] += 1
        header = [''] + self.classes
        data = []
        for x in range(len(self.classes)):
            data.append([self.classes[x]] + matrix[x].tolist())
        return data, header

    def conv2d(self, x, filter_count, filter_size):
        W = tf.Variable(tf.truncated_normal([filter_size, filter_size, x.shape[3].value, filter_count], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([filter_count], stddev=0.1))
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b)
        return tf.nn.relu(conv)

    def conv2d_135(self, x, filter_count):
        one_filter = tf.Variable(tf.truncated_normal([1, 1, x.shape[3].value, filter_count], stddev=0.1))
        one_conv = tf.nn.conv2d(x, one_filter, strides=[1, 1, 1, 1], padding='SAME')

        three_filter = tf.Variable(tf.truncated_normal([3, 3, x.shape[3].value, filter_count], stddev=0.1))
        three_conv = tf.nn.conv2d(x, three_filter, strides=[1, 1, 1, 1], padding='SAME')

        five_filter = tf.Variable(tf.truncated_normal([5, 5, x.shape[3].value, filter_count], stddev=0.1))
        five_conv = tf.nn.conv2d(x, five_filter, strides=[1, 1, 1, 1], padding='SAME')

        conv = tf.concat([one_conv, three_conv, five_conv], axis=3)
        b = tf.Variable(tf.truncated_normal([3 * filter_count], stddev=0.1))
        conv = tf.nn.bias_add(conv, b)
        return tf.nn.relu(conv)

    def conv2d_135_reduction(self, x, filter_count):
        one_filter = tf.Variable(tf.truncated_normal([1, 1, x.shape[3].value, filter_count], stddev=0.1))
        one_conv = tf.nn.conv2d(x, one_filter, strides=[1, 1, 1, 1], padding='SAME')

        three_conv_1 = self.conv2d(x, 1, 1)
        three_filter = tf.Variable(tf.truncated_normal([3, 3, 1, filter_count], stddev=0.1))
        three_conv = tf.nn.conv2d(three_conv_1, three_filter, strides=[1, 1, 1, 1], padding='SAME')

        five_conv_1 = self.conv2d(x, 1, 1)
        five_filter = tf.Variable(tf.truncated_normal([5, 5, 1, filter_count], stddev=0.1))
        five_conv = tf.nn.conv2d(five_conv_1, five_filter, strides=[1, 1, 1, 1], padding='SAME')

        conv = tf.concat([one_conv, three_conv, five_conv], axis=3)
        b = tf.Variable(tf.truncated_normal([3 * filter_count], stddev=0.1))
        conv = tf.nn.bias_add(conv, b)
        return tf.nn.relu(conv)

    def fc(self, x, length):
        W = tf.Variable(tf.truncated_normal([x.shape[1].value, length], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([length], stddev=0.1))
        return tf.matmul(x, W) + b

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def run(self):
        # Save the log file
        log = open(os.path.join(self.directory, 'log.txt'), 'w')
        log.write("Epochs: {}\n".format(self.epochs))
        log.write("Batch Size: {}\n".format(self.batch_size))
        log.write("Learning Rate: {}\n".format(self.learning_rate))
        log.write("Model Definition:\n{}\n\n".format(self.model_definition))
        log.write("Fully Connected Size: {}\n".format(self.final_layer_size))
        log.write("\n")

        X = tf.placeholder(tf.float32, [None] + list(self.input_shape))
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

        last_layer = X
        network = []
        summary = ''
        for line in self.model_definition.split("\n"):
            parts = line.split(',')
            layer = None
            if 'conv2d' == parts[0]:
                layer = self.conv2d(last_layer, int(parts[1]), int(parts[2]))
                summary += "{}\n".format(layer.shape)
            elif 'conv2d_135' == parts[0]:
                layer = self.conv2d_135(last_layer, int(parts[1]))
                summary += "{}\n".format(layer.shape)
            elif 'conv2d_135_reduction' == parts[0]:
                layer = self.conv2d_135_reduction(last_layer, int(parts[1]))
                summary += "{}\n".format(layer.shape)
            elif 'max_pool' == parts[0]:
                layer = self.max_pool_2x2(last_layer)
            else:
                self.feedback.emit('Trainer', 'Unknown layer type: [{}]'.format(line))
                log.write('Invalid model definition.')
                log.close()
                return
            network.append(layer)
            last_layer = layer

        length = last_layer.shape[1].value * last_layer.shape[2].value * last_layer.shape[3].value
        flat = tf.reshape(last_layer, [-1, length])

        fc_1 = tf.nn.relu(self.fc(flat, self.final_layer_size))
        summary += "{}\n".format(fc_1.shape)

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        dropout = tf.nn.dropout(fc_1, keep_prob)

        layer_out = self.fc(dropout, self.n_classes)
        summary += "{}\n".format(layer_out.shape)
        log.write("Layers Summary:\n{}\n\n".format(summary))

        prediction = tf.nn.softmax(layer_out, name='prediction')
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=layer_out))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_batch = int(len(self.training_data) / self.batch_size) + 1
            batch_x = []
            batch_y = []
            for i in range(total_batch):
                batch_x.append(np.array(self.training_data[self.batch_size * i:self.batch_size * (i + 1)]))
                batch_y.append(np.array(self.training_labels[self.batch_size * i:self.batch_size * (i + 1)]))
            for epoch in range(self.epochs):
                avg_loss = 0
                avg_acc = 0
                for b in range(total_batch):
                    train_op.run(feed_dict={X: batch_x[b], Y: batch_y[b], keep_prob: 0.5})
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x[i], Y: batch_y[i], keep_prob: 1.0})
                    avg_loss += loss / total_batch
                    avg_acc += acc / total_batch
                # loss, train_accuracy = sess.run([loss_op, accuracy], feed_dict={X: self.training_data, Y: self.training_labels, keep_prob: 1.0})
                # message = "Epoch {} loss: {:.4f} accuracy: {:.3f}".format(epoch, loss, train_accuracy)
                message = 'Epoch: {} Avg Batch [loss: {:.4f}  acc: {:.3f}]'.format(epoch, avg_loss, avg_acc)
                self.feedback.emit('Train', message)
                log.write(message + "\n")

                self.progress.emit(epoch + 1)
                if self.stop:
                    self.feedback.emit('Train', 'Training interrupted.')
                    log.write("Training interrupted.\n")
                    break

            pred_train = sess.run(prediction, feed_dict={X: self.training_data, keep_prob: 1.0})
            pred_validation = sess.run(prediction, feed_dict={X: self.validation_data, keep_prob: 1.0})

            message = "Train Acc: {:.5f}".format(sess.run(accuracy, feed_dict={X: self.training_data, Y: self.training_labels, keep_prob: 1.0}))
            self.feedback.emit('Train', message)
            log.write("\n" + message + "\n")

            data, header = self.confusion_matrix(pred_train, self.training_labels)
            output = tabulate(data, headers=header, tablefmt='orgtbl')
            log.write("Training data confusion matrix\n")
            log.write(output + "\n\n")

            message = "Validation Acc: {:.5f}".format(sess.run(accuracy, feed_dict={X: self.validation_data, Y: self.validation_labels, keep_prob: 1.0}))
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
