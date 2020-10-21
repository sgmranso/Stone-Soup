# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import multivariate_normal
from scipy.stats import poisson as poisson

from .base import Hypothesiser
from ..base import Property
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleProbabilityHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
from ..types.state import GaussianState
from ..types.state import GaussianStateWithExistence
from ..predictor import Predictor
from ..updater import Updater
from ..functions import gm_reduce_single


class PDAHypothesiser(Hypothesiser):
    """Hypothesiser based on Probabilistic Data Association (PDA)

    Generate track predictions at detection times and calculate probabilities
    for all prediction-detection pairs for single prediction and multiple
    detections.
    """

    predictor: Predictor = Property(doc="Predict tracks to detection times")
    updater: Updater = Property(doc="Updater used to get measurement prediction")
    clutter_spatial_density: float = Property(
        doc="Spatial density of clutter - tied to probability of false detection")
    surveillence_volume: float = Property(
        doc="Surveillence volume - required to determine the expected quantity of clutter.")
    prob_detect: Probability = Property(
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_gate: Probability = Property(
        default=Probability(0.95),
        doc="Gate Probability - prob. gate contains true measurement "
            "if detected")

    def hypothesise(self, track, detections, timestamp):
        r"""Evaluate and return all track association hypotheses.

        For a given track and a set of N detections, return a
        MultipleHypothesis with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated probability.
        Probabilities are assumed to be exhaustive (sum to 1) and mutually
        exclusive (two detections cannot be the correct association at the
        same time).

        Detection 0: missed detection, none of the detections are associated
        with the track.
        Detection :math:`i, i \in {1...N}`: detection i is associated
        with the track.

        The probabilities for these detections are calculated as follow:

        .. math::

          \beta_i(k) = \begin{cases}
                \frac{\mathcal{L}_{i}(k)}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=1,...,m(k) \\
                \frac{1-P_{D}P_{G}}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=0
                \end{cases}

        where

        .. math::

          \mathcal{L}_{i}(k) = \frac{\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),
          S(k)]P_{D}}{\lambda}

        :math:`\lambda` is the clutter density

        :math:`P_{D}` is the detection probability

        :math:`P_{G}` is the gate probability

        :math:`\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),S(k)]` is the likelihood
        ratio of the measurement :math:`z_{i}(k)` originating from the track
        target rather than the clutter.

        NOTE: Since all probabilities have the same denominator and are
        normalized later, the denominator can be discarded.

        References:

        [1] "The Probabilistic Data Association Filter: Estimation in the
        Presence of Measurement Origin Uncertainty" -
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5338565

        [2] "Robotics 2 Data Association" (Lecture notes) -
        http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/pdfs/rob2-15-dataassociation.pdf

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on
        detections: :class:`list`
            A list of :class:`~Detection` objects, representing the available
            detections.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleProbabilityHypothesis` objects

        """

        hypotheses = list()
        prediction = self.predictor.predict(track, timestamp=timestamp)# state prediction
        probability = Probability(
            (1 - self.prob_detect*self.prob_gate
            ) * poisson.pmf( # poisson distribution pdf
            len(detections), # all detections are clutter
            self.clutter_spatial_density * self.surveillence_volume # expected number of clutter detections
            )
        ) # misdetection hypothesis
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability
            )
        ) # append

        # True detection hypotheses
        for detection in detections:
            # Compute measurement prediction and probability measure
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model)
            # Calculate difference before to handle custom types (mean defaults to zero)
            # This is required as log pdf coverts arrays to floats
            #logpdf = multivariate_normal.logpdf(
            #    detection.state_vector-measurement_prediction.state_vector,
            #    cov=measurement_prediction.covar
            #)
            #pdf = Probability(logpdf)
            err = detection.state_vector-measurement_prediction.state_vector
            pdf = math.exp(
                -0.5*err.T@np.matrix.getI(measurement_prediction.covar)@err
            )/math.sqrt(
                pow(2*math.pi,len(detection.state_vector))*np.linalg.det(measurement_prediction.covar)
            )
            probability = Probability(
                pdf * self.prob_detect * poisson.pmf( # poisson distribution pdf
                len(detections)-1, # all but one detections are clutter
                self.clutter_spatial_density * self.surveillence_volume # expected number of clutter detections
                )/self.clutter_spatial_density
            )
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction)
            ) # True detection hypothesis

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)


class IPDAHypothesiser(Hypothesiser):
    """Hypothesiser based on Integrated Probabilistic Data Association (IPDA)

    Generate surviving and newly born track predictions at detection times and
    calculate probabilities for all prediction-detection pairs for single
    prediction and multiple detections.
    """

    predictor: Predictor = Property(doc="Predict tracks to detection times")
    updater: Updater = Property(doc="Updater used to get measurement prediction")
    clutter_spatial_density: float = Property(
        doc="Spatial density of clutter - tied to probability of false detection")
    surveillence_volume: float = Property(
        doc="Surveillence volume - required to determine the expected quantity of clutter.")
    prob_detect: Probability = Property(
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_gate: Probability = Property(
        default=Probability(0.95),
        doc="Gate Probability - prob. gate contains true measurement "
            "if detected")

    def hypothesise(self, track, detections, timestamp):
        r"""Evaluate and return all track association hypotheses.

        For a given track and a set of N detections, return a
        MultipleHypothesis with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated probability.
        Probabilities are assumed to be exhaustive (sum to 1) and mutually
        exclusive (two detections cannot be the correct association at the
        same time).

        Detection 0: missed detection, none of the detections are associated
        with the track.
        Detection :math:`i, i \in {1...N}`: detection i is associated
        with the track.

        The probabilities for these detections are calculated as follow:

        .. math::

          \beta_i(k) = \begin{cases}
                \frac{\mathcal{L}_{i}(k)}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=1,...,m(k) \\
                \frac{1-P_{D}P_{G}}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=0
                \end{cases}

        where

        .. math::

          \mathcal{L}_{i}(k) = \frac{\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),
          S(k)]P_{D}}{\lambda}

        :math:`\lambda` is the clutter density

        :math:`P_{D}` is the detection probability

        :math:`P_{G}` is the gate probability

        :math:`\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),S(k)]` is the likelihood
        ratio of the measurement :math:`z_{i}(k)` originating from the track
        target rather than the clutter.

        NOTE: Since all probabilities have the same denominator and are
        normalized later, the denominator can be discarded.

        References:

        [1] "The Probabilistic Data Association Filter: Estimation in the
        Presence of Measurement Origin Uncertainty" -
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5338565

        [2] "Robotics 2 Data Association" (Lecture notes) -
        http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/pdfs/rob2-15-dataassociation.pdf

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on
        detections: :class:`list`
            A list of :class:`~Detection` objects, representing the available
            detections.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleProbabilityHypothesis` objects.
            ####################################################################################################
            ###                                     IMPORTANT                                                ###
            ### WEIGHTS ARE RETURNED UNNORMALISED FOR THE PURPOSES OF DETERMINING THE EXISTENCE PROBABILITY. ###
            ####################################################################################################

        """

        hypotheses = list()

        # Common state & measurement prediction
        prediction = self.predictor.predict(
            track,
            timestamp=timestamp
        )

        # Missed detection hypothesis
        probability = Probability( # probability class
                ((1 - self.prob_detect*self.prob_gate) * prediction.probability # target exists, but is misdetected 
                ) * poisson.pmf( # poisson distribution pdf
                len(detections), # all detections are clutter
                self.clutter_spatial_density*self.surveillence_volume # expected number of clutter detections
                )
        )
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability
            )
        )

        # True detection hypotheses
        for detection in detections:
            measurement_prediction = self.updater.predict_measurement( # measurement predictor object
                prediction, # state prediction
                detection.measurement_model # measurement model (done here, since multisensor detections need different models)
            ) # measurement prediction
            # Calculate difference before to handle custom types (mean defaults to zero)
            # This is required as log pdf coverts arrays to floats
            #logpdf = multivariate_normal.logpdf(
            #    detection.state_vector-measurement_prediction.state_vector,
            #    cov=measurement_prediction.covar
            #)
            #pdf = Probability(logpdf)
            err = detection.state_vector-measurement_prediction.state_vector
            pdf = math.exp(
                -0.5*err.T@np.matrix.getI(measurement_prediction.covar)@err
            )/math.sqrt(
                pow(2*math.pi,len(detection.state_vector))*np.linalg.det(measurement_prediction.covar)
            )
            probability = Probability(
                (pdf * self.prob_detect * prediction.p_exist # target exists and is detected 
                ) * poisson.pmf( # poisson distribution pdf
                len(detections) - 1, # all but one detection is clutter
                self.clutter_spatial_density*self.surveillence_volume # expected number of clutter detections 
                )/self.clutter_spatial_density
            )

            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction
                )
            )
        
        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)