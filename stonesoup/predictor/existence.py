# -*- coding: utf-8 -*-

import math
import numpy as np

from .kalman import KalmanPredictor
from ._utils import predict_lru_cache
from ..base import Property
from ..models.base import LinearModel
from ..models.control import ControlModel
from ..models.control.linear import LinearControlModel
from ..models.transition import TransitionModel
from ..models.transition.linear import LinearGaussianTransitionModel
from ..types.prediction import GaussianStateWithExistencePrediction
from ..types.numeric import Probability


class KalmanPredictorWithExistence(KalmanPredictor):
    r"""A predictor class which forms the basis for the family of Kalman
    predictors with additional existence probability functionality. This
    class also serves as the (specific) Kalman Filter
    :class:`~.Predictor` class. Here

    .. math::

      f_k( \mathbf{x}_{k-1}) = F_k \mathbf{x}_{k-1},  \ b_k( \mathbf{u}_k) =
      B_k \mathbf{u}_k \ \mathrm{and} \ \mathbf{\nu}_k \sim \mathcal{N}(0,Q_k)

    and

    .. math::
      
      p (e_{k}|Z_{k-1}) = \sum_{e_{k-1}} p (e_{k}|e_{k-1}) p (e_{k-1}|Z_{k-1}),     e_{k} = E_{k}.

    Notes
    -----
    In the Kalman filter, transition and control models must be linear.

    Raises
    ------
    ValueError
        If no :class:`~.TransitionModel` is specified.

    """

    transition_model: LinearGaussianTransitionModel = Property(
        doc="The transition model to be used.")
    control_model: LinearControlModel = Property(
        default=None,
        doc="The control model to be used. Default `None` where the predictor "
            "will create a zero-effect linear :class:`~.ControlModel`.")
    target_existence_interval: float = Property(doc="Expected target presence; "
        "influences the birth probability.")
    target_absence_interval: float = Property(doc="Expected target absence;"
        "influences the death probability.")

    # This attribute tells the :meth:`predict()` method what type of prediction to return
    _prediction_class = GaussianStateWithExistencePrediction

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If no control model insert a linear zero-effect one
        # TODO: Think about whether it's more efficient to leave this out
        if self.control_model is None:
            ndims = self.transition_model.ndim_state
            self.control_model = LinearControlModel(ndims, [],
                                                    np.zeros([ndims, 1]),
                                                    np.zeros([ndims, ndims]),
                                                    np.zeros([ndims, ndims]))

    def _transition_matrix(self, **kwargs):
        """Return the transition matrix

        Parameters
        ----------
        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`numpy.ndarray`
            The transition matrix, :math:`F_k`

        """
        return self.transition_model.matrix(**kwargs)

    def _transition_function(self, prior, **kwargs):
        r"""Applies the linear transition function to a single vector in the
        absence of a control input, returns a single predicted state.

        Parameters
        ----------
        prior : :class:`~.GaussianState`
            The prior state, :math:`\mathbf{x}_{k-1}`

        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.State`
            The predicted state

        """
        return self.transition_model.matrix(**kwargs) @ prior.state_vector

    @property
    def _control_matrix(self):
        r"""Convenience function which returns the control matrix

        Returns
        -------
        : :class:`numpy.ndarray`
            control matrix, :math:`B_k`

        """
        return self.control_model.matrix()

    def _predict_over_interval(self, prior, timestamp, **kwargs):
        """Private function to get the prediction interval (or None)

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state

        timestamp : :class:`datetime.datetime`, optional
            The (current) timestamp

        Returns
        -------
        : :class:`datetime.timedelta`
            time interval to predict over

        """

        # Deal with undefined timestamps
        if timestamp is None or prior.timestamp is None:
            predict_over_interval = None
        else:
            predict_over_interval = timestamp - prior.timestamp

        return predict_over_interval

    def _predicted_covariance(self, prior, predict_over_interval, **kwargs):
        """Private function to return the predicted covariance. Useful in that
        it can be overwritten in children.

        Parameters
        ----------
        prior : :class:`~.GaussianState`
            The prior class
        predict_over_interval : :class`~.timedelta`

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The predicted covariance matrix

        """
        prior_cov = prior.covar
        trans_m = self._transition_matrix(
            prior=prior,
            time_interval=predict_over_interval,
            **kwargs
        )
        trans_cov = self.transition_model.covar(
            time_interval=predict_over_interval,
            **kwargs
        )

        # As this is Kalman-like, the control model must be capable of
        # returning a control matrix (B)
        ctrl_mat = self._control_matrix
        ctrl_noi = self.control_model.control_noise

        return trans_m @ prior_cov @ trans_m.T + trans_cov + ctrl_mat @ ctrl_noi @ ctrl_mat.T

    def _birth_probability(self, predict_over_interval, **kwargs):
        """Private function to return the birth probability.

        Parameters
        ----------
        predict_over_interval : :class`~.timedelta`
        target_existence_interval : :class`~.timedelta`

        Returns
        ----------
        : :class:`~.Probability`
            The expected target birth rate (i.e. birth probability).

        """
        dt = predict_over_interval.seconds+predict_over_interval.microseconds/1000000
        return Probability(1-math.exp(-dt/self.target_existence_interval))  # usually the same as _death_probability

    def _target_born_probability(self, prior, predict_over_interval, **kwargs):
        """Private function to return the probability a target is born.

        Parameters
        ----------
        predict_over_interval : :class`~.timedelta`
        target_existence_interval : :class`~.timedelta`

        Returns
        ----------
        : :class:`~.Probability`
            The expected target birth rate (i.e. birth probability).

        """
        p_exist = prior.p_exist
        p_birth = self._birth_probability(
            predict_over_interval=predict_over_interval,
            **kwargs
        )
        
        return Probability(p_birth*(1-p_exist))
    
    def _death_probability(self, predict_over_interval, **kwargs):
        """Private function to return the death probability.

        Parameters
        ----------
        predict_over_interval : :class`~.timedelta`
        target_absence_interval : :class`~.timedelta`

        Returns
        ----------
        : :class:`~.Probability`
            The expected target death rate (i.e. death probability).

        """
        dt = predict_over_interval.seconds+predict_over_interval.microseconds/1000000
        return 1-math.exp(-dt/self.target_absence_interval)  # usually the same as _birth_probability
    
    def _target_survive_probability(self, prior, predict_over_interval, **kwargs):
        """Private function to return the target survival probability.

        Parameters
        ----------
        p_exist : :class`~.Probability`
            Previous posterior existence probability.

        Returns
        ----------
        : :class:`~.Probability`
            The probability an old target survives.

        """
        p_exist = prior.p_exist
        p_death = self._death_probability(
            predict_over_interval=predict_over_interval,
            **kwargs)
        
        return Probability((1-p_death)*p_exist)

    def _predicted_existence_probability(self, prior, predict_over_interval, **kwargs):
        """Private function to return the predicted existence probability. Useful in that
        it can be overwritten in children.

        Parameters
        ----------
        prior : :class:`~.GaussianStateWithExistence`
            The prior class
        predict_over_interval : :class`~.timedelta`

        Returns
        -------
        : :class:`~.Probability`
            The predicted existence probability.

        """
        p_born = self._target_born_probability(
            prior=prior,
            predict_over_interval=predict_over_interval,
            **kwargs)
        p_survive = self._target_survive_probability(
            prior=prior,
            predict_over_interval=predict_over_interval,
            **kwargs)

        return Probability(p_born+p_survive)

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.State`
            :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`, optional
            :math:`k`
        **kwargs :
            These are passed, via :meth:`~.KalmanFilter.transition_function` to
            :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.GaussianStatePrediction`
            :math:`\mathbf{x}_{k|k-1}`, the predicted state and the predicted
            state covariance :math:`P_{k|k-1}`

        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(
            prior=prior,
            timestamp=timestamp,
            **kwargs
        )

        # Prediction of the mean
        state_prediction = self._transition_function(
            prior=prior,
            time_interval=predict_over_interval,
            **kwargs
        ) + self.control_model.control_input()

        # Prediction of the covariance
        covariance_prediction = self._predicted_covariance(
            prior=prior,
            predict_over_interval=predict_over_interval,
            **kwargs
        )

        # Prediction of the existence probability
        existence_probability_prediction = self._predicted_existence_probability(
            prior=prior,
            predict_over_interval=predict_over_interval,
            **kwargs
        )

        # And return the state in the correct form
        return self._prediction_class(
            state_prediction,
            covariance_prediction,
            existence_probability_prediction,
            timestamp=timestamp
        )

