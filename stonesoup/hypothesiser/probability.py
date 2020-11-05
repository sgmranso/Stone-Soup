# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import poisson, multivariate_normal

from .base import Hypothesiser
from ..base import Property
from ..functions import gm_reduce_single
from ..models.existence import BirthModel, DeathModel
from ..types.array import StateVectors
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleProbabilityHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
from ..predictor import Predictor
from ..types.prediction import GaussianStateWithExistencePrediction
from ..updater import Updater


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
    surveillance_volume: float = Property(
        doc="Surveillance volume - required to determine the expected quantity of clutter.")
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
        prediction = self.predictor.predict(track, timestamp=timestamp)  # state prediction
        probability = Probability(
            (1 - self.prob_detect * self.prob_gate
             ) * poisson.pmf(  # poisson distribution pdf
                len(detections),  # all detections are clutter
                self.clutter_spatial_density * self.surveillance_volume  # expected number of clutter detections
            )
        )  # mis-detection hypothesis
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability
            )
        )  # append

        # True detection hypotheses
        for detection in detections:
            # Compute measurement prediction and probability measure
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model)
            # Calculate difference before to handle custom types (mean defaults to zero)
            # This is required as log pdf coverts arrays to floats
            log_pdf = multivariate_normal.logpdf(
               detection.state_vector-measurement_prediction.state_vector,
               cov=measurement_prediction.covar
            )
            pdf = Probability(log_pdf, log_value='true')
            probability = Probability(
                pdf * self.prob_detect * poisson.pmf(  # poisson distribution pdf
                    len(detections) - 1,  # all but one detections are clutter
                    self.clutter_spatial_density * self.surveillance_volume  # expected number of clutter detections
                ) / self.clutter_spatial_density
            )
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction)
            )  # True detection hypothesis

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
    surveillance_volume: float = Property(
        doc="Surveillance volume - required to determine the expected quantity of clutter.")
    prob_detect: Probability = Property(
        default=Probability(0.9),
        doc="Target Detection Probability")
    prob_gate: Probability = Property(
        default=Probability(0.99),
        doc="Gate Probability - prob. gate contains true measurement "
            "if detected")
    merge_strategy: str = Property(
        default='a priori',
        doc='The hypothesis merging strategy: may be either "a priori" or "a posteriori".')
    birth_model: BirthModel = Property(
        doc='Birth model',
        default=None)
    death_model: DeathModel = Property(
        doc='Death model',
        default=None)

    def hypothesise(self, track, detections, timestamp):
        r"""Evaluate and return all track association hypotheses.

        Assuming the existence (i.e. :math:`e_{k}=E_{k}`) of a given track
        and a set of N detections, return a MultipleHypothesis with N+1
        detections (first detection is a 'MissedDetection') conditional on
        target existence and an absence hypothesis (i.e. :math:`e_{k}=E_{k}`)
        conditional on target non-existence each with an associated probability.
        Probabilities are assumed to be exhaustive (sum to 1) and mutually
        exclusive (two detections cannot be the correct association at the
        same time).

        Detection E_{k}, 0: missed detection, none of the detections are associated
        with the track.
        Detection :math:`E_{k}, i, i \in {1...N}`: track is assumed to exist (i.e.
        :math:`e_{k}=E_{k}`) and detection i is associated with the track.
        Absent \bar{E}_{k}: absent target

        The probabilities for these detections are calculated as follow:

        .. math::

          \beta_{k}^{e_{k},i} =
            \begin{cases}
                \text{Poisson}(M_{k}; \lambda V_{k}) p(\bar{D}_{k},\bar{G}_{k}), \quad e_{k}=E_{k}, i=0 \\
                \text{Poisson}(M_{k}-1; \lambda V_{k}) p(D_{k},G_{k})
                    \mathcal{L}_{k}^{i}, \quad e_{k}=E_{k}, i=1,...,m(k) \\
                \text{Poisson}(M_{k}; \lambda V_{k}), \quad e_{k}=\bar{E}_{k},
            \end{cases}

        where

        .. math::

          \mathcal{L}_{k}^{i} = \frac{\mathcal{N}(z_{k}^{i};\hat{z}_{k|k-1},\hat{S}_{k|k-1})}{p(G_{k}) \lambda}

        :math:`\lambda` is the clutter density

        :math:`p(D_{k})` is the detection probability (i.e. :math:`d_{k}=D_{k}` where :math:`p(\bar{D}_{k})=1-p(D_{k})`)

        :math:`p(G_{k})` is the gate probability (i.e. :math:`g_{k}=G_{k}` where :math:`p(\bar{G}_{k})=1-p(G_{k})`)

        :math:`\mathcal{N}(z_{k}^{i};\hat{z}_{k|k-1},\hat{S}_{k|k-1})` is the likelihood of the measurement
        :math:`z_{i}(k)` originating from the track target rather than the clutter.

        NOTE: Since all probabilities have the same denominator and are
        normalized later, the denominator can be discarded.

        References:

        [1] "Integrated expected likelihood particle filters" -
        https://www.doi.org/10.23919/FUSION45008.2020.9190387

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
        """

        hypotheses = list()

        # Common state & measurement prediction
        survive_prediction = self.predictor.predict(
            track,
            timestamp=timestamp
        )
        p_exist = survive_prediction.p_exist
        survive_prediction.p_exist = ((1 - self.death_model.death_probability) * track.p_exist)

        birth_prediction = GaussianStateWithExistencePrediction(
            self.birth_model.birth_density.mean,  # initial mean
            self.birth_model.birth_density.covar,  # initial covariance
            (self.birth_model.birth_probability * (1 - track.p_exist)),  # target born probability
            timestamp=timestamp
        )  # birth hypothesis

        priors = [survive_prediction, birth_prediction]  # store survival and birth states
        if self.merge_strategy == 'a priori':
            mus = StateVectors([state.state_vector for state in priors])  # priori state vectors
            Sigmas = np.stack([state.covar for state in priors], axis=2)  # priori state covariances
            weights = np.array([
                survive_prediction.p_exist/p_exist,  # survival weight
                birth_prediction.p_exist/p_exist  # birth weight
            ])  # priori weights
            mu_merge, Sigma_merge = gm_reduce_single(
                mus,  # a priori birth and survival means
                Sigmas,  # a priori birth and survival covariances
                weights  # a priori existence probability weights
            )  # merge priori densities
            prediction = GaussianStateWithExistencePrediction(
                mu_merge,  # a priori merged mean
                Sigma_merge,  # a priori merged covariance
                p_exist,  # a priori existence probability
                timestamp=timestamp
            )  # new merged a priori track object

            # Missed detection hypothesis
            probability = Probability(  # probability class
                ((1 - self.prob_detect * self.prob_gate) * prediction.p_exist  # target exists, but is mis-detected
                 ) * poisson.pmf(  # poisson distribution pdf
                    len(detections),  # all detections are clutter
                    self.clutter_spatial_density * self.surveillance_volume  # expected number of clutter detections
                )
            )
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    MissedDetection(timestamp=timestamp),
                    probability,
                )
            )

            # Target absence hypothesis
            prob_not_exist = Probability(
                poisson.pmf(  # Poisson pdf
                    len(detections),  # all detections assumed clutter
                    self.clutter_spatial_density * self.surveillance_volume  # mean quantity of clutter
                ) * (1-prediction.p_exist)  # a priori absence probability
            )

            hypotheses.append(
                SingleProbabilityHypothesis(
                    None,
                    None,
                    Probability(prob_not_exist)
                )
            )

            # Detection hypotheses
            for detection in detections:
                measurement_prediction = self.updater.predict_measurement(  # measurement predictor object
                    prediction,  # state prediction
                    detection.measurement_model  # measurement model (multi-sensor detections need different models)
                )  # measurement prediction
                # Calculate difference before to handle custom types (mean defaults to zero)
                # This is required as log pdf coverts arrays to floats
                log_pdf = multivariate_normal.logpdf(
                   np.array(detection.state_vector-measurement_prediction.state_vector).T,
                   cov=measurement_prediction.covar
                )  # this breaks for some reason
                pdf = Probability(log_pdf, log_value='true')  # see above; replaced below
                probability = Probability(
                    (pdf * self.prob_detect * prediction.p_exist  # target exists and is detected
                     ) * poisson.pmf(  # poisson distribution pdf
                        len(detections) - 1,  # all but one detection is clutter
                        self.clutter_spatial_density * self.surveillance_volume  # expected number of clutter detections
                    )/self.clutter_spatial_density
                )
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prediction,
                        detection,
                        probability,
                        measurement_prediction
                    )
                )
        elif self.merge_strategy == 'a posteriori':
            for prior in priors:  # store survival and birth states
                # Missed detection hypothesis
                probability = Probability(  # probability class
                    ((1 - self.prob_detect * self.prob_gate) * prior.p_exist  # target born, but mis-detected
                     ) * poisson.pmf(  # poisson distribution pdf
                        len(detections),  # all detections are clutter
                        self.clutter_spatial_density * self.surveillance_volume  # expected number of clutter detections
                    )
                )
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prior,
                        MissedDetection(timestamp=timestamp),
                        probability,
                    )
                )

                # Not existing hypothesis
                prob_not_exist = Probability(
                    poisson.pmf(
                        len(detections),  # all detections are clutter
                        self.clutter_spatial_density * self.surveillance_volume) * (
                            1 - prior.p_exist
                    )
                )
                hypotheses.append(SingleProbabilityHypothesis(
                    None,
                    None,
                    Probability(prob_not_exist)
                ))
                # Detection hypotheses
                for detection in detections:
                    measurement_prediction = self.updater.predict_measurement(  # measurement predictor object
                        prior,  # state prediction
                        detection.measurement_model  # measurement model (multi-sensor detections need different models)
                    )  # measurement prediction
                    # Calculate difference before to handle custom types (mean defaults to zero)
                    # This is required as log pdf coverts arrays to floats
                    log_pdf = multivariate_normal.logpdf(
                       detection.state_vector-measurement_prediction.state_vector,
                       cov=measurement_prediction.covar
                    )  # log pdf for detection and prediction
                    pdf = Probability(log_pdf, log_value='true')  # pdf
                    probability = Probability(
                        (pdf * self.prob_detect * prior.p_exist  # target exists and is detected
                         ) * poisson.pmf(  # poisson distribution pdf
                            len(detections) - 1,  # all but one detection is clutter
                            self.clutter_spatial_density * self.surveillance_volume  # mean clutter detections
                        ) / self.clutter_spatial_density
                    )
                    hypotheses.append(
                        SingleProbabilityHypothesis(
                            prior,
                            detection,
                            probability,
                            measurement_prediction
                        )
                    )
        else:
            raise ValueError('Invalid hypothesis management strategy. Use "a priori" or "a posteriori" merging.')

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)
