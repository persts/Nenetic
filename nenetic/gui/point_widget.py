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
from PyQt5 import QtCore, QtGui, QtWidgets, uic

WIDGET, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'point_widget.ui'))


class PointWidget(QtWidgets.QWidget, WIDGET):

    def __init__(self, canvas, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.canvas = canvas

        self.pushButtonAddClass.clicked.connect(self.add_class)
        self.pushButtonRemoveClass.clicked.connect(self.remove_class)
        self.pushButtonSave.clicked.connect(self.save)
        self.pushButtonLoadPoints.clicked.connect(self.load)
        self.pushButtonReset.clicked.connect(self.reset)
        self.pushButtonExport.clicked.connect(self.export)

        self.tableWidgetClasses.verticalHeader().setVisible(False)
        self.tableWidgetClasses.horizontalHeader().setMinimumSectionSize(1)
        self.tableWidgetClasses.horizontalHeader().setStretchLastSection(False)
        self.tableWidgetClasses.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tableWidgetClasses.setColumnWidth(1, 30)
        self.tableWidgetClasses.cellClicked.connect(self.cell_clicked)
        self.tableWidgetClasses.cellChanged.connect(self.cell_changed)
        self.tableWidgetClasses.selectionModel().selectionChanged.connect(self.selection_changed)

        self.checkBoxDisplayPoints.toggled.connect(self.display_points)
        self.canvas.image_loaded.connect(self.image_loaded)
        self.canvas.update_point_count.connect(self.update_point_count)
        self.canvas.points_loaded.connect(self.display_classes)

        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)
        self.reset_model()
        self.treeView.doubleClicked.connect(self.double_click)

        self.spinBoxPointRadius.valueChanged.connect(self.canvas.set_point_radius)

    def add_class(self):
        class_name, ok = QtWidgets.QInputDialog.getText(self, 'New Class', 'Class Name')
        if ok:
            self.canvas.add_class(class_name)
            self.display_classes()
            self.display_count_tree()

    def display_points(self, display):
        self.canvas.toggle_points(display=display)

    def double_click(self, model_index):
        item = self.model.itemFromIndex(model_index)
        if item.isSelectable():
            path = os.path.join(self.canvas.directory, item.text())
            self.canvas.load_image(path)

    def cell_changed(self, row, column):
        if column == 0:
            old_class = self.canvas.classes[row]
            new_class = self.tableWidgetClasses.item(row, column).text()
            if old_class != new_class:
                self.tableWidgetClasses.selectionModel().clear()
                self.canvas.rename_class(old_class, new_class)
                self.display_classes()
                self.display_count_tree()

    def cell_clicked(self, row, column):
        if column == 1:
            color = QtWidgets.QColorDialog.getColor()
            if color.isValid():
                self.canvas.colors[self.canvas.classes[row]] = color
                item = QtWidgets.QTableWidgetItem()
                icon = QtGui.QPixmap(20, 20)
                icon.fill(color)
                item.setData(QtCore.Qt.DecorationRole, icon)
                self.tableWidgetClasses.setItem(row, 1, item)

    def display_classes(self):
        self.tableWidgetClasses.setRowCount(len(self.canvas.classes))
        row = 0
        for class_name in self.canvas.classes:
            item = QtWidgets.QTableWidgetItem(class_name)
            self.tableWidgetClasses.setItem(row, 0, item)

            item = QtWidgets.QTableWidgetItem()
            icon = QtGui.QPixmap(20, 20)
            icon.fill(self.canvas.colors[class_name])
            item.setData(QtCore.Qt.DecorationRole, icon)
            self.tableWidgetClasses.setItem(row, 1, item)
            row += 1
        self.tableWidgetClasses.selectionModel().clear()

    def display_count_tree(self):
        self.reset_model()
        for image in self.canvas.points:
            image_item = QtGui.QStandardItem(image)
            image_item.setEditable(False)
            self.model.appendRow(image_item)
            if image == self.canvas.current_image_name:
                font = image_item.font()
                font.setBold(True)
                image_item.setFont(font)
                self.treeView.setExpanded(image_item.index(), True)

            for class_name in self.canvas.classes:
                class_item = QtGui.QStandardItem(class_name)
                class_item.setEditable(False)
                class_item.setSelectable(False)
                class_count = QtGui.QStandardItem('0')
                if class_name in self.canvas.points[image]:
                    class_count = QtGui.QStandardItem(str(len(self.canvas.points[image][class_name])))
                class_count.setEditable(False)
                class_count.setSelectable(False)
                image_item.appendRow([class_item, class_count])

    def export(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Export Points To Directory', self.canvas.directory)
        if directory != '':
            self.canvas.export_points(directory)

    def image_loaded(self, directory, file_name):
        self.tableWidgetClasses.selectionModel().clear()
        self.display_count_tree()

    def load(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Training Points', self.canvas.directory, 'Point Files (*.pnt)')
        if file_name[0] != '':
            self.canvas.load_points(file_name[0])

    def reset(self):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setWindowTitle('Warning')
        msgBox.setText('You are about to clear all training data')
        msgBox.setInformativeText('Do you want to continue?')
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        response = msgBox.exec()
        if response == QtWidgets.QMessageBox.Ok:
            self.canvas.reset()
            self.display_classes()
            self.display_count_tree()

    def reset_model(self):
        self.model.clear()
        self.model.setColumnCount(2)
        self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Image')
        self.model.setHeaderData(1, QtCore.Qt.Horizontal, 'Count')
        self.treeView.setExpandsOnDoubleClick(False)
        self.treeView.header().setStretchLastSection(False)
        self.treeView.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

    def remove_class(self):
        indexes = self.tableWidgetClasses.selectedIndexes()
        if len(indexes) > 0:
            class_name = self.canvas.classes[indexes[0].row()]
            msgBox = QtWidgets.QMessageBox()
            msgBox.setWindowTitle('Warning')
            msgBox.setText('You are about to remove class [{}] '.format(class_name))
            msgBox.setInformativeText('Do you want to continue?')
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
            msgBox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            response = msgBox.exec()
            if response == QtWidgets.QMessageBox.Ok:
                self.canvas.remove_class(class_name)
                self.display_classes()
                self.display_count_tree()

    def save(self):
        file_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Training Points', os.path.join(self.canvas.directory, 'untitled.pnt'), 'Point Files (*.pnt)')
        if file_name[0] != '':
            self.canvas.save_points(file_name[0])

    def selection_changed(self, selected, deselected):
        if len(selected.indexes()) > 0:
            self.canvas.set_current_class(selected.indexes()[0].row())
        else:
            self.canvas.set_current_class(None)

    def update_point_count(self, image_name, class_name, class_count):
        items = self.model.findItems(image_name)
        if len(items) == 0:
            self.display_count_tree()
        else:
            items[0].child(self.canvas.classes.index(class_name), 1).setText(str(class_count))
