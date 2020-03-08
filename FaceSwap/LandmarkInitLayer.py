from lasagne.layers import Layer


class LandmarkInitLayer(Layer):
    def __init__(self, increments, init_landmarks, **kwargs):
        super().__init__(increments, **kwargs)

        self.init_landmarks = init_landmarks.flatten()

    @staticmethod
    def get_output_shape_for(input_shape):
        return input_shape

    def get_output_for(self, input):
        output = input + self.init_landmarks

        return output
