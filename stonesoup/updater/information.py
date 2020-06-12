# -*- coding: utf-8 -*-

import numpy as np
from functools import lru_cache

from ..base import Property
from ..types.prediction import InformationMeasurementPrediction
from ..types.update import InformationStateUpdate
from ..models.measurement.linear import LinearGaussian
from ..updater.kalman import KalmanUpdater


class InfoFilterUpdater(KalmanUpdater):
    r"""A class to implement the update of Information filter.

    The Information Filter update class inherits from the Kalman filter updater. assume
    :math:`h(\mathbf{x}) = H \mathbf{x}` with additive noise :math:`\sigma = \mathcal{N}(0,R)`.
    Daughter classes can overwrite to specify a more general measurement model
    :math:`h(\mathbf{x})`.

    :meth:`update` first calls :meth:`predict_measurement` function which
    proceeds by calculating the predicted measurement, innovation covariance
    and measurement cross-covariance,

    .. math::

        \mathbf{I}_{k|k-1} = H^{T}_k R^{-1}_k H

        \mathbf{i}_{k|k-1} = H^{T}_k R^{-1}_k \mathbf{z}_{k}

        \Sigma = \[G^T M G + Q^{-1} \]

    where :math:`y_{k|k-1}` is the predicted information state and G is the "information
    observation" matrix which is assumed to be set to the identity matrix.
    :meth:`predict_measurement` returns a
    :class:`~.InformationMeasurementPrediction`. The information prediction gain (analogous to the
    Kalman gain) is then calculated as,

    .. math::

        \Omega_k = M_k G_k \Sigma^{-1}_k

    and the posterior information state mean and information matrix are,

    .. math::

        y_{k|k} = y_{k|k-1} + i_k

        Y_{k|k} = Y_{k|k-1} + I_k

    These are returned as a :class:`~.GaussianStateUpdate` object.
    """

    # TODO: at present this will throw an error if a measurement model is not
    # TODO: specified in either individual measurements or the Updater object
    measurement_model = Property(
        LinearGaussian, default=None,
        doc="A linear Gaussian measurement model. This need not be defined if "
            "a measurement model is provided in the measurement. If no model "
            "specified on construction, or in the measurement, then error "
            "will be thrown.")

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Predict the information theoretic quantity implied by the predicted state mean and
        predicted measurement. Function is provided for completeness the
        :meth:`~.predict_information_state' method is recommended for data association when
        implementing the information filter in a multi-target scenario.

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function` and
            :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`InformationMeasurementPrediction`
            The information-theoretic measurement prediction, :math:`\mathbf{\upsilon}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        # See equations 294 and 295 in Hugh Durrant-Whyte document
        #      pred_meas = measurement_model.function(predicted_state, **kwargs) # Returns a vector
        #      in measurement space

        # projection matrix from measurement space to information space.
        proj_matrix = hh.T @ np.linalg.inv(measurement_model.covar())

        # information theoretic quantity - project measurement space to info theoretic space.
        pred_info_meas = hh @ np.linalg.inv(
            predicted_state.info_matrix) @ predicted_state.state_vector

        # innov_cov = hh@predicted_state.covar@hh.T + measurement_model.covar()
        innov_cov = hh @ predicted_state.info_matrix @ hh.T + measurement_model.covar()
        # meas_cross_cov = predicted_state.info_matrix @ hh.T

        return InformationMeasurementPrediction(pred_info_meas, innov_cov,
                                                predicted_state.timestamp, proj_matrix=proj_matrix)

    def update(self, hypothesis, force_symmetric_covariance=False, **kwargs):
        r"""The Information filter update (estimate) method. Given a hypothesised association
        between a predicted information state or predicted measurement and an actual measurement,
        calculate the posterior information state.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        force_symmetric_covariance : :obj:`bool`, optional
            A flag to force the output covariance matrix to be symmetric by way
            of a simple geometric combination of the matrix and transpose.
            Default is `False`
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.GaussianStateUpdate`
            The posterior state Gaussian with mean :math:`\mathbf{x}_{k|k}` and
            covariance :math:`P_{x|x}`

        """

        # Get the predicted state out of the hypothesis
        predicted_state = hypothesis.prediction

        # #If there is no measurement prediction in the hypothesis then do the
        # #measurement prediction (and attach it back to the hypothesis).
        # if hypothesis.measurement_prediction is None:
        #     # Get the measurement model out of the measurement if it's there.
        #     # If not, use the one native to the updater (which might still be
        #     # none)
        #     measurement_model = hypothesis.measurement.measurement_model
        #     measurement_model = self._check_measurement_model(
        #         measurement_model)
        #
        #     # Attach the measurement prediction to the hypothesis
        #     hypothesis.measurement_prediction = self.predict_measurement(
        #         predicted_state, measurement_model=measurement_model, **kwargs)

        measurement_model = hypothesis.measurement.measurement_model
        measurement_model = self._check_measurement_model(
            measurement_model)

        # Attach the measurement prediction to the hypothesis
        hypothesis.measurement_prediction = self.predict_measurement(
            predicted_state, measurement_model=measurement_model, **kwargs)

        y = hypothesis.prediction
        H = measurement_model.matrix()
        R = measurement_model.noise_covar

        posterior_mean = y.state_vector + H.T @ np.linalg.inv(R) @ hypothesis.measurement.state_vector
        posterior_information_matrix = hypothesis.prediction.info_matrix +\
                                       H.T @ np.linalg.inv(R) @ H

        # Complete the calculation of the posterior
        # This isn't optimised

        if force_symmetric_covariance:
            posterior_information_matrix = \
                (posterior_information_matrix + posterior_information_matrix.T)/2

        return InformationStateUpdate(
            posterior_mean,
            posterior_information_matrix,
            hypothesis,
            hypothesis.measurement.timestamp)