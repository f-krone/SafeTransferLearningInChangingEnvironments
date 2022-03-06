from math import floor
def conv_output_shape_3d(h_w_d, kernel_size=1, stride=1, pad=0):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if type(stride) is not tuple:
            stride = (stride, stride, stride)
        h = floor( (h_w_d[0] + (2 * pad) - kernel_size[0])/ stride[0]) + 1
        w = floor( (h_w_d[1] + (2 * pad) - kernel_size[1])/ stride[1]) + 1
        d = floor( (h_w_d[2] + (2 * pad) - kernel_size[2])/ stride[2]) + 1
        return h, w, d

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor( (h_w[0] + (2 * pad) - kernel_size[0])/ stride) + 1
        w = floor( (h_w[1] + (2 * pad) - kernel_size[1])/ stride) + 1
        return h, w