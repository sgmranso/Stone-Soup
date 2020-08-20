# -*- coding: utf-8 -*-

from ...base import Property
from ..base import NonLinearModel


class SimpleHarmonicMotion(NonLinearModel):
    """
    A simple harmonic transition model. For details of the simple harmonic
    oscillator, see XX

    TODO: Implement this

    """

    amplitude = Property(float, doc="The amplitude of the simple harmonic "
                                    "oscillator")
    phase = Property(float)  # TODO: make this an angle i.e. 0<phi<2*pi

    def function(self, state_vector, noise=None, **kwargs):
        """

        :param state_vector:
        :param noise:
        :param kwargs:
        :return:
        """