import numpy as np
from scipy.signal import convolve2d as c2d


class Generator():
    def __init__(self):
        self.layers = []
        self.stack = np.array(None)

    def generate(self, image, layer_definitions=[], pad=0):
        self.max_value = 255
        self.stack = (image / self.max_value).astype(np.float32)
        bands = np.split(np.int32(image), image.shape[2], axis=2)
        if image.shape[2] == 3:
            self.layers = ['band_1', 'band_2', 'band_3']
        else:
            self.layers = ['band_1', 'band_2', 'band_3', 'band_4']
        for layer_def in layer_definitions:
            if layer_def['name'] == 'average':
                kernels = layer_def['kernels']
                solid_kernel = layer_def['solid_kernel']
                for size in range(3, (kernels * 2) + 3, 2):
                    kernel = np.ones((size, size))
                    if not solid_kernel:
                        kernel[1:size - 1, 1:size - 1] = 0
                    kernel = kernel / np.sum(kernel)
                    for band in range(image.shape[2]):
                        b = c2d(image[:, :, band], kernel, mode='same')
                        b = (b / self.max_value).astype(np.float32)
                        self.stack = np.dstack((self.stack, b))
                        self.layers.append('band_{}_avg_{}'.format(band + 1, size))
            elif layer_def['name'] == 'gli':
                denom = np.clip(2 * bands[1] + bands[0] + bands[2], 1, None)
                gli = (((2 * bands[1] - bands[0] - bands[2]) / denom) + 1.0) / 2.0
                self.stack = np.dstack((self.stack, gli.astype(np.float32)))
                self.layers.append('gli')
            elif layer_def['name'] == 'lightness':
                maximum = np.maximum(bands[0], bands[1])
                maximum = np.maximum(maximum, bands[2])
                minimum = np.minimum(bands[0], bands[1])
                minimum = np.minimum(minimum, bands[2])
                lightness = ((maximum + minimum) / 2) / self.max_value
                self.stack = np.dstack((self.stack, lightness.astype(np.float32)))
                self.layers.append('lightness')
            elif layer_def['name'] == 'luminosity':
                luminosity = (0.21 * bands[0] + 0.72 * bands[1] + 0.07 * bands[2]) / self.max_value
                self.stack = np.dstack((self.stack, luminosity.astype(np.float32)))
                self.layers.append('luminosity')
            elif layer_def['name'] == 'rgb_average':
                average = ((bands[0] + bands[1] + bands[2]) / 3) / self.max_value
                self.stack = np.dstack((self.stack, average.astype(np.float32)))
                self.layers.append('rgb_average')
            elif layer_def['name'] == 'vari':
                denom = bands[1] + bands[0] - bands[2]
                denom[denom == 0.0] = 1
                vari = (((bands[1] - bands[0]) / denom) + 1.0) / 2.0
                self.stack = np.dstack((self.stack, vari.astype(np.float32)))
                self.layers.append('vari')
            elif layer_def['name'] == 'vndvi':
                denom = np.clip(bands[1] + bands[0], 1, None)
                vndvi = (((bands[1] - bands[0]) / denom) + 1.0) / 2.0
                self.stack = np.dstack((self.stack, vndvi.astype(np.float32)))
                self.layers.append('vndvi')
            else:
                pass
        if pad > 0:
            self.stack = np.pad(self.stack, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
        return self.stack, self.layers
