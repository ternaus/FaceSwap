from lasagne.layers import MergeLayer
import theano
from theano import tensor as T


class LandmarkTransformLayer(MergeLayer):
    def __init__(self, landmarks, transform_params, inverse=False, **kwargs):
        super().__init__([landmarks, transform_params], **kwargs)

        self.inverse = inverse

    def affine_transform_helper(self, landmarks, transform):
        A = T.zeros((2, 2))

        A = T.set_subtensor(A[0, 0], transform[0])
        A = T.set_subtensor(A[0, 1], transform[1])
        A = T.set_subtensor(A[1, 0], transform[2])
        A = T.set_subtensor(A[1, 1], transform[3])
        t = transform[4:6]

        if self.inverse:
            A = T.nlinalg.matrix_inverse(A)
            t = T.dot(-t, A)

        output = (T.dot(landmarks.reshape((-1, 2)), A) + t).flatten()
        return output

    @staticmethod
    def get_output_shape_for(input_shapes):
        output_shape = list(input_shapes[0])
        return tuple(output_shape)

    def get_output_for(self, inputs):
        outImgs, _ = theano.scan(self.affine_transform_helper, inputs)

        return outImgs
