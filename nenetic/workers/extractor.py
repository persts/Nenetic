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

from nenetic.extractors import *  # noqa: F403, F401


class Extractor(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    feedback = QtCore.pyqtSignal(str, str)

    def __init__(self, extractor_name, packaged_points, file_name, directory, kwargs={}):
        QtCore.QThread.__init__(self)
        self.file_name = file_name
        self.extractor_name = extractor_name
        self.kwargs = kwargs
        self.directory = directory
        self.points = packaged_points

    def run(self):
        extractor = globals()[self.extractor_name](**self.kwargs)
        extractor.progress.connect(self.pass_progress)
        extractor.feedback.connect(self.pass_feedback)
        extractor.load_points(self.points, self.directory)

        extractor.extract()
        self.sleep(1)  # Sleep to let gui catch up
        extractor.save(self.file_name)

    def pass_feedback(self, tool, message):
        self.feedback.emit(tool, message)

    def pass_progress(self, progress):
        self.progress.emit(progress)
