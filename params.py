class params:

    def __init__(self):
        self.MINS=20
        self.lonMin, self.lonMax = 127.070, 127.135
        self.latMin, self.latMax = 37.485, 37.525
        self.step = 64

        self.lonTick = ((self.lonMax)-(self.lonMin))/self.step
        self.latTick = ((self.latMax)-(self.latMin))/self.step

    # xTicks = [lonMin + i*lonTick for i in range(self.step)]
    # yTicks = [latMin + i*latTick for i in range(self.step)]