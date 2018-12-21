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
import time
import pickle
import numpy as np
from PyQt5 import QtWidgets, uic

from nenetic.workers import Extractor
from nenetic.workers import FcTrainer
from nenetic.workers import ConvTrainer
from nenetic.workers import Classifier

CLASS_DIALOG, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'toolkit_widget.ui'))


class ToolkitWidget(QtWidgets.QDialog, CLASS_DIALOG):

    def __init__(self, canvas, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.canvas = canvas
        self.training_data = None
        self.directory = None

        self.extractor = None
        self.fc_trainer = None
        self.conv_trainer = None
        self.classifier = Classifier()
        self.classifier.feedback.connect(self.log)
        self.classifier.update.connect(self.canvas.update_classified_image)
        self.classifier.finished.connect(self.enable_action_buttons)

        self.progress_max = 0
        self.pushButtonExtract.clicked.connect(self.extract_training_data)

        self.pushButtonFcTrainingData.clicked.connect(self.load_fc_training_data)
        self.pushButtonTrainFcModel.clicked.connect(self.train_fc_model)
        self.pushButtonStopFcTraining.clicked.connect(self.stop_fc_training)

        self.pushButtonConvTrainingData.clicked.connect(self.load_conv_training_data)
        self.pushButtonTrainConvModel.clicked.connect(self.train_conv_model)
        self.pushButtonStopConvTraining.clicked.connect(self.stop_conv_training)

        self.pushButtonLoadModel.clicked.connect(self.load_model)
        self.pushButtonClassify.clicked.connect(self.classify_image)
        self.pushButtonSaveClassification.clicked.connect(self.save_classification)
        self.pushButtonStopClassification.clicked.connect(self.stop_classification)
        self.pushButtonLoadClassification.clicked.connect(self.load_classification)
        self.checkBoxShowClassification.stateChanged.connect(self.show_classification)
        self.horizontalSliderOpacity.valueChanged.connect(self.canvas.set_opacity)

        self.radioButtonVector.clicked.connect(self.update_region_size_controls)
        self.radioButtonRaster.clicked.connect(self.update_region_size_controls)

        self.canvas.image_loaded.connect(self.image_loaded)

    def classify_image(self):
        if self.canvas.base_image is not None:
            self.checkBoxShowClassification.setChecked(True)
            array = np.array(self.canvas.base_image)
            self.progressBar.setValue(0)
            self.progressBar.setRange(0, array.shape[0])
            self.classifier.image = array
            self.classifier.threshold = self.doubleSpinBoxConfidence.value()
            self.pushButtonStopClassification.setEnabled(True)
            self.disable_action_buttons()
            self.classifier.start()

    def disable_action_buttons(self):
        self.pushButtonExtract.setEnabled(False)
        self.pushButtonFcTrainingData.setEnabled(False)
        self.pushButtonTrainFcModel.setEnabled(False)
        self.pushButtonConvTrainingData.setEnabled(False)
        self.pushButtonTrainConvModel.setEnabled(False)

        self.pushButtonClassify.setEnabled(False)
        self.pushButtonSaveClassification.setEnabled(False)
        self.pushButtonLoadModel.setEnabled(False)

    def enable_action_buttons(self):
        self.pushButtonExtract.setEnabled(True)
        self.pushButtonFcTrainingData.setEnabled(True)
        self.pushButtonTrainFcModel.setEnabled(True)
        self.pushButtonConvTrainingData.setEnabled(True)
        self.pushButtonTrainConvModel.setEnabled(True)

        self.pushButtonClassify.setEnabled(True)
        self.pushButtonSaveClassification.setEnabled(True)
        self.pushButtonLoadModel.setEnabled(True)

        self.pushButtonStopFcTraining.setEnabled(False)
        self.pushButtonStopConvTraining.setEnabled(False)
        self.pushButtonStopClassification.setEnabled(False)
        self.classifier.stop = False

        self.extractor = None
        self.fc_trainer = None
        self.conv_trainer = None

    def extract_training_data(self):
        self.directory = self.canvas.directory
        package, point_count = self.canvas.package_points()
        if point_count > 0:
            if self.checkBoxJson.isChecked():
                file_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Training Data', os.path.join(self.directory, 'untitled.json'), 'Point Files (*.json)')
            else:
                file_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Training Data', os.path.join(self.directory, 'untitled.p'), 'Point Files (*.p)')
            if file_name[0] is not '':
                self.disable_action_buttons()

                self.progress_max = point_count
                self.progressBar.setValue(0)
                self.progressBar.setRange(0, 0)

                pad = self.spinBoxRegionSize.value() // 2
                layer_definitions = self.layer_definitions()
                if self.radioButtonVector.isChecked():
                    extractor_name = 'Neighborhood'
                else:
                    extractor_name = 'Region'

                self.extractor = Extractor(extractor_name, layer_definitions, pad, package, file_name[0], self.canvas.directory)
                self.extractor.progress.connect(self.update_progress)
                self.extractor.feedback.connect(self.log)
                self.extractor.finished.connect(self.enable_action_buttons)
                self.extractor.start()

        else:
            self.log('Extractor', 'Zero training points')

    def image_loaded(self, directory, image_name):
        self.checkBoxShowClassification.setChecked(False)

    def layer_definitions(self):
        definitions = []
        if self.checkBoxAverage.isChecked():
            definitions.append({'name': 'average', 'kernels': self.spinBoxKernels.value(), 'solid_kernel': self.checkBoxSolidKernel.isChecked()})
        if self.checkBoxVndvi.isChecked():
            definitions.append({'name': 'vndvi'})
        if self.checkBoxGli.isChecked():
            definitions.append({'name': 'gli'})
        if self.checkBoxLightness.isChecked():
            definitions.append({'name': 'lightness'})
        if self.checkBoxLuminosity.isChecked():
            definitions.append({'name': 'luminosity'})
        if self.checkBoxRgbAverage.isChecked():
            definitions.append({'name': 'rgb_average'})
        if self.checkBoxVari.isChecked():
            definitions.append({'name': 'vari'})
        return definitions

    def load_classification(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Classified Image', self.directory, 'PNG (*.png)')
        if file_name[0] is not '':
            self.classifier.load_classified_image(file_name[0])

    def load_model(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Model', self.directory, 'Model Metadata (*.meta)')
        if file_name[0] is not '':
            self.classifier = Classifier(file_name[0])
            self.classifier.progress.connect(self.progressBar.setValue)
            self.classifier.feedback.connect(self.log)
            self.classifier.update.connect(self.canvas.update_classified_image)
            self.classifier.finished.connect(self.enable_action_buttons)

    def load_conv_training_data(self):
        self.log('Trainer', 'Loading training data...')
        self.progressBar.setValue(0)
        self.progressBar.setRange(0, 0)
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Training Data', self.directory, 'Point Files (*.json *.p)')
        if file_name[0] is not '':
            if file_name[0][-4:].lower() == 'json':
                file = open(file_name[0], 'r')
                self.training_data = json.load(file)
                file.close()
                self.training_data['data'] = np.array(self.training_data['data'])
            else:
                file = open(file_name[0], 'rb')
                self.training_data = pickle.load(file)
                file.close()
            self.log('Trainer', 'Done.')
            if 'type' in self.training_data['extractor'] and self.training_data['extractor']['type'] == 'raster':
                total_points = self.training_data['data'].shape[0]
                image_size = self.training_data['data'][0].shape
                num_classes = len(self.training_data['labels'][0])
                self.labelImageSize.setText("({}, {}, {})".format(image_size[0], image_size[1], image_size[2]))
                self.labelNumberClassesConv.setText("{}".format(num_classes))
                self.labelTotalPointsConv.setText("{}".format(total_points))
            else:
                self.training_data = None
                self.log('Trainer', 'Wrong training data, format is vector.')
                self.labelImageSize.setText("0")
                self.labelNumberClassesConv.setText("0")
                self.labelTotalPointsConv.setText("0")
        else:
            self.log('Trainer', 'Canceled.')
        self.progressBar.setRange(0, 1)

    def load_fc_training_data(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Training Data', self.directory, 'Point Files (*.json *.p)')
        if file_name[0] is not '':
            if file_name[0][-4:].lower() == 'json':
                file = open(file_name[0], 'r')
                self.training_data = json.load(file)
                file.close()
                self.training_data['data'] = np.array(self.training_data['data'])
            else:
                file = open(file_name[0], 'rb')
                self.training_data = pickle.load(file)
                file.close()
            if 'type' in self.training_data['extractor'] and self.training_data['extractor']['type'] == 'raster':
                self.training_data = None
                self.log('Trainer', 'Wrong training data, format is raster.')
                self.labelVectorLength.setText("0")
                self.labelNumberClasses.setText("0")
                self.labelTotalPoints.setText("0")
            else:
                total_points = self.training_data['data'].shape[0]
                vector_length = self.training_data['data'].shape[1]
                num_classes = len(self.training_data['labels'][0])
                self.labelVectorLength.setText("{}".format(vector_length))
                self.labelNumberClasses.setText("{}".format(num_classes))
                self.labelTotalPoints.setText("{}".format(total_points))

    def log(self, tool, message):
        text = "[{}]({}) {}".format(time.strftime('%H:%M:%S'), tool, message)
        self.textBrowserConsole.append(text)

    def save_classification(self):
        if self.classifier is not None:
            file_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Classified Image', os.path.join(self.canvas.directory, 'untitled.png'), 'PNG (*.png)')
            if file_name[0] is not '':
                self.classifier.save_classification(file_name[0])

    def show_classification(self):
        self.canvas.toggle_classification(self.checkBoxShowClassification.isChecked())

    def stop_classification(self):
        self.classifier.stop = True

    def stop_conv_training(self):
        self.conv_trainer.stop = True

    def stop_fc_training(self):
        self.fc_trainer.stop = True

    def train_conv_model(self):
        if self.training_data is not None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Save Model To Directory', self.directory)
            if directory != '':
                params = {}
                params['epochs'] = self.spinBoxEpochsConv.value()
                params['learning_rate'] = self.doubleSpinBoxLearningRateConv.value()
                params['batch_size'] = self.spinBoxBatchSizeConv.value()
                params['model'] = self.textBrowserConvModel.toPlainText()
                params['fc_size'] = self.spinBoxFinalLayerConv.value()
                params['validation_split'] = self.doubleSpinBoxSplitConv.value()
                self.conv_trainer = ConvTrainer(self.training_data, directory, params)

                self.progressBar.setValue(0)
                self.progressBar.setRange(0, self.spinBoxEpochsConv.value())
                self.conv_trainer.progress.connect(self.update_progress)
                self.conv_trainer.feedback.connect(self.log)
                self.conv_trainer.finished.connect(self.enable_action_buttons)
                self.pushButtonStopConvTraining.setEnabled(True)
                self.disable_action_buttons()
                self.conv_trainer.start()

    def train_fc_model(self):
        if self.training_data is not None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Save Model To Directory', self.directory)
            if directory != '':
                params = {}
                params['epochs'] = self.spinBoxEpochs.value()
                params['learning_rate'] = self.doubleSpinBoxLearningRate.value()
                params['batch_size'] = self.spinBoxBatchSize.value()
                params['l1_hidden_nodes'] = self.spinBoxL1.value()
                params['l2_hidden_nodes'] = self.spinBoxL2.value()
                params['validation_split'] = self.doubleSpinBoxSplit.value()
                self.fc_trainer = FcTrainer(self.training_data, directory, params)

                self.progressBar.setValue(0)
                self.progressBar.setRange(0, self.spinBoxEpochs.value())
                self.fc_trainer.progress.connect(self.update_progress)
                self.fc_trainer.feedback.connect(self.log)
                self.fc_trainer.finished.connect(self.enable_action_buttons)
                self.pushButtonStopFcTraining.setEnabled(True)
                self.disable_action_buttons()
                self.fc_trainer.start()

    def update_progress(self, value):
        if self.progressBar.minimum() == self.progressBar.maximum():
            self.progressBar.setRange(0, self.progress_max)
        self.progressBar.setValue(value)

    def update_region_size_controls(self):
        if self.radioButtonVector.isChecked():
            self.spinBoxRegionSize.setMinimum(1)
            self.spinBoxRegionSize.setValue(1)
        else:
            self.spinBoxRegionSize.setMinimum(21)
            self.spinBoxRegionSize.setValue(31)
