# Nenetic
The Neural Network Image Classifier (Nenetic) is an open source tool written in Python to label image pixels with discrete classes to create products such as land cover maps. The user interface is designed to facilitate a workflow that involves selecting training data locations, extracting training data using original image pixel data and computed features, building models, and classifying images. The current version works with 3-band images such as those acquired from typical digital cameras.


![Screen Shot](/doc/NeneticTraining.png)


## Installation

### Dependencies
Nenetic is being developed on Ubuntu 18.04 with the following libraries:

* PyQt5 (5.10.1)
* TKinter (3.6.7)
* Pillow (5.4.1)
* Numpy (1.15.4)
* Scipy (1.2.0)
* Tabulate (0.8.2)
* Psutil (5.4.8)
* Tensorflow

Install GUI libraries:

``` bash
sudo apt install python3-pyqt5 python3-tk
```
Install pip3 and install / upgrade dependencies:

```bash
sudo apt install python3-pip
sudo -H pip3 install --upgrade pillow
sudo -H pip3 install numpy
sudo -H pip3 install scipy
sudo -H pip3 install tabulate
sudo -H pip3 install psutil
```

For detailed steps to install TensorFlow, follow the [TensorFlow installation instructions](https://www.tensorflow.org/install/). A typical user can install Tensorflow using one of the following commands:
``` bash
# For CPU
sudo -H pip3 install tensorflow
# For GPU
sudo -H pip3 install tensorflow-gpu
```

## Launching Nenetic
```bash
git clone https://github.com/persts/Nenetic
cd Nenetic
python3 main.py
```