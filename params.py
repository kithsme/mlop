class params:

    def __init__(self):
        self.MINS=20
        self.lonMin, self.lonMax = 127.070, 127.135
        self.latMin, self.latMax = 37.485, 37.525
        self.step = 64

        self.lonTick = ((self.lonMax)-(self.lonMin))/self.step
        self.latTick = ((self.latMax)-(self.latMin))/self.step

        self.training_set_size = 0.7
        self.test_set_size = 1.0 - self.training_set_size

        self.mini_batch_size = 75

        self.conv1_filter_size = 6
        self.conv1_stride = 3
        self.conv1_output_size = 32
        self.conv2_filter_size = 3
        self.conv2_stride = 2
        self.conv2_output_size = 64
        self.conv3_filter_size = 3
        self.conv3_stride = 2
        self.conv3_output_size = 128
        self.conv4_filter_size = 3
        self.conv4_stride = 2
        self.conv4_output_size = 256
        self.conv5_filter_size = 3
        self.conv5_stride = 2
        self.conv5_output_size = 256
        self.fc1_output_size = 1024