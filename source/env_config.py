class EnvConfig:
    def __init__(self,
                 pixels_per_meter,
                 screen_width,
                 screen_height):
        # this class functions as a class that can set up the environment
        self.pixels_per_meter = pixels_per_meter
        self.screen_width = screen_width
        self.screen_height = screen_height

        # some colors
        self.white = (255, 255, 255)
        self.gray = (100, 100, 100)
        self.black = (0, 0, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        