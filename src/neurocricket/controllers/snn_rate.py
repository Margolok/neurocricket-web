from neurocricket.neuroscience.snn_network import SNNNetwork


class SNNController:
    def __init__(self, mode, settings, seed):
        self.network = SNNNetwork(mode, settings, seed)

    def step(self, reading):
        return self.network.advance(reading.proximity)
