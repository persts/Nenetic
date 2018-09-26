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
from PyQt5 import QtWidgets, QtCore


class CentralGraphicsView(QtWidgets.QGraphicsView):
    add_point = QtCore.pyqtSignal(QtCore.QPointF)
    load_image = QtCore.pyqtSignal(QtCore.QUrl)
    region_selected = QtCore.pyqtSignal(QtCore.QRectF)
    delete_selection = QtCore.pyqtSignal()
    relabel_selection = QtCore.pyqtSignal()
    toggle_points = QtCore.pyqtSignal()
    toggle_classification = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        QtWidgets.QGraphicsView.__init__(self, parent)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.shift = False
        self.ctrl = False
        self.alt = False
        self.delay = 0
        self.setViewportUpdateMode(0)

    def enterEvent(self, event):
        self.setFocus()

    def image_loaded(self, directory, file_name):
        self.resetTransform()
        self.fitInView(self.scene().itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        self.setSceneRect(self.scene().itemsBoundingRect())

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Alt:
            self.alt = True
        elif event.key() == QtCore.Qt.Key_Control:
            self.ctrl = True
        elif event.key() == QtCore.Qt.Key_Shift:
            self.shift = True
        elif event.key() == QtCore.Qt.Key_Delete:
            self.delete_selection.emit()
        elif event.key() == QtCore.Qt.Key_R:
            self.relabel_selection.emit()
        elif event.key() == QtCore.Qt.Key_D:
            self.toggle_points.emit()
        elif event.key() == QtCore.Qt.Key_C:
            self.toggle_classification.emit()

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Alt:
            self.alt = False
        elif event.key() == QtCore.Qt.Key_Control:
            self.ctrl = False
        elif event.key() == QtCore.Qt.Key_Shift:
            self.shift = False

    def mouseMoveEvent(self, event):
        if self.alt:
            self.delay += 1
        else:
            QtWidgets.QGraphicsView.mouseMoveEvent(self, event)
        if self.delay == 7:
            self.delay = 0
            self.add_point.emit(self.mapToScene(event.pos()))

    def mousePressEvent(self, event):
        if self.ctrl:
            self.add_point.emit(self.mapToScene(event.pos()))
        elif self.shift:
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
            QtWidgets.QGraphicsView.mousePressEvent(self, event)
        else:
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            QtWidgets.QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        if self.dragMode() == QtWidgets.QGraphicsView.RubberBandDrag:
            rect = self.rubberBandRect()
            self.region_selected.emit(self.mapToScene(rect).boundingRect())
            QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    def dragEnterEvent(self, event):
        event.setAccepted(True)

    def dragMoveEvent(self, event):
        pass

    def dropEvent(self, event):
        if len(event.mimeData().urls()) > 0:
            self.load_image.emit(event.mimeData().urls()[0])

    def wheelEvent(self, event):
        if len(self.scene().items()) > 0:
            if event.angleDelta().y() > 0:
                self.scale(1.1, 1.1)
            else:
                self.scale(0.9, 0.9)
