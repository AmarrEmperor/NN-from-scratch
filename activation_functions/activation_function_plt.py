import numpy as np
from manim import *

class ActivationFunctionPlot(Scene):
    def construct(self):
        axes = Axes(x_range=[-5, 5], y_range=[-1, 1], axis_config={"color": BLUE})
        labels = axes.get_axis_labels(Tex("x"), Tex("f(x)"))
        
        # Define the activation functions
        sigmoid = axes.plot(lambda x: 1 / (1 + np.exp(-x)), x_range=[-5, 5], color=YELLOW)
        relu = axes.plot(lambda x: max(0, x), x_range=[-5, 5], color=GREEN)
        tanh = axes.plot(lambda x: np.tanh(x), x_range=[-5, 5], color=RED)

        # Create the scene
        self.play(Create(axes), Create(labels))
        self.play(Create(sigmoid), run_time=2)
        self.wait(1)
        self.play(Create(relu), run_time=2)
        self.wait(1)
        self.play(Create(tanh), run_time=2)
        self.wait(1)