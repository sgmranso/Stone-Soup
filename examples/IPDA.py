#!/usr/bin/env python
# coding: utf-8

# Some general imports and set up
import numpy
from datetime import datetime, timedelta
from scipy.stats import poisson
from math import sqrt, pi, cos, sin
from matplotlib import pyplot
from matplotlib.patches import Ellipse
# stonesoup imports
from stonesoup.functions import gm_reduce_single   # merges posterior estimates using mixture reduction
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity  # transition
from stonesoup.models.measurement.linear import LinearGaussian  # linear measurement model
from stonesoup.models.existence import BirthModel, DeathModel  # birth and death models
from stonesoup.hypothesiser.probability import IPDAHypothesiser  # import hypothesiser
from stonesoup.dataassociator.probability import IPDA  # import integrated probabilistic data association
from stonesoup.predictor.existence import KalmanPredictorWithExistence  # import Kalman predictor
from stonesoup.updater.existence import KalmanUpdaterWithExistence  # import Kalman updater
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState  # container for groundtruth
from stonesoup.types.detection import TrueDetection  # container for target detections
from stonesoup.types.detection import Clutter  # container for clutter detections
from stonesoup.types.track import Track  # container for track object
from stonesoup.types.array import StateVectors   # container for state vectors
from stonesoup.types.state import GaussianStateWithExistence  # container for previous posteriori estimate
from stonesoup.types.prediction import GaussianStateWithExistencePrediction   # container for priori estimate
from stonesoup.types.update import GaussianStateWithExistenceUpdate   # container for posteriori estimate

# simulation variables
start_time = datetime.now()
T = 100  # temporal length
dt = 1  # timestep length
dim = 2  # position dimensions
sigma_b = numpy.array([300, 10])  # birth covariance coefficient
sigma_x = 2  # process noise coefficient, q
sigma_z = 3  # measurement noise coefficient, r
p_birth = 0.04  # target birth probability
p_death = 0.04   # target death probability
rmax = 1000   # maximum surveillance range
V = pi*pow(rmax, 2)   # surveillance volume
lmbd = 1*pow(10, -6)  # poisson spatial intensity
pD = 0.9  # probability of detection
pG = 1 - numpy.finfo(float).eps  # gating probability: nearly 1
gamma = 9  # gating scalar in std
merge_strategy = 'a priori'   # IPDA hypothesis merging strategy: choose between 'a priori' and 'a posteriori'
# Gaussian birth density with existence
mu = numpy.array(numpy.repeat([0], 4))  # zero-mean state
Sigma = numpy.diag(numpy.tile(numpy.power(sigma_b, 2), 2))  # birth density
p_exist = 0  # existence probability
transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(numpy.power(sigma_x, 2)), 
     ConstantVelocity(numpy.power(sigma_x, 2)))
)
measurement_model = LinearGaussian(
    4,  # Number of state dimensions (position and velocity in 2D)
    (0, 2),  # Mapping measurement vector index to state index
    numpy.diag(numpy.repeat(numpy.power([sigma_z], 2), 2))
)
birth_model = BirthModel(p_birth)
p_birth = birth_model.birth_probability
death_model = DeathModel(p_death)
p_death = death_model.death_probability
target_start = numpy.random.multivariate_normal(mu, Sigma)
state = GroundTruthState(target_start, timestamp=start_time)
states = GroundTruthPath([state])
old_time = start_time
for k in numpy.arange(dt, T, dt):
    new_time = old_time + timedelta(seconds=dt)
    state = GroundTruthState(
        transition_model.function(
            state, 
            noise=True, 
            time_interval=timedelta(seconds=dt)
        ), timestamp=new_time
    )
    states.append(state)
    old_time = new_time
detection_sets = []
for state in states:
    detection_set = set()
    if numpy.random.rand() <= pD:
        measurement = measurement_model.function(state, noise=True)
        if numpy.linalg.norm(measurement) < rmax:
            detection_set.add(TrueDetection(
                state_vector=measurement, 
                groundtruth_path=state, 
                timestamp=state.timestamp
            ))
    
    Nc = numpy.random.poisson(lmbd*V)  # random sample clutter quantity from Poisson
    for m in range(1, Nc):
        rho = rmax*sqrt(numpy.random.random())
        theta = 2*pi*numpy.random.random()
        x = rho*cos(theta)
        y = rho*sin(theta)
        detection_set.add(Clutter(
            numpy.array([[x], [y]]), timestamp=state.timestamp
        ))
        
    detection_sets.append(detection_set)
predictor = KalmanPredictorWithExistence(
    transition_model=transition_model, 
    target_existence_interval=T/2, 
    target_absence_interval=T/2
)  # state prediction from transition, birth and death models
updater = KalmanUpdaterWithExistence(measurement_model)
data_associator = IPDA(  # IPDA data association object
    hypothesiser=IPDAHypothesiser(  # IPDA hypothesiser
        predictor,  # predictor object
        updater,  # updater object
        lmbd,  # clutter density
        V,  # surveillance volume
        pD,  # detection probability
        pG,  # gating probability
    )
)
track = GaussianStateWithExistence(
    mu, 
    Sigma, 
    p_exist, 
    timestamp=start_time
)  # create prior
tracks = Track([track])  # append to track
old_time = start_time  # initiate timestep
for k, detections in enumerate(detection_sets):  # for all detection at time k
    new_time = old_time + timedelta(seconds=dt)  # update new time
    p_exist = p_birth*(1-p_exist)+(1-p_death)*p_exist  # predicted existence probability
    if merge_strategy == 'a priori':  # method for combining a priori hypotheses before detections
        birth_track = Track([GaussianStateWithExistence(
            mu,  # initial mean
            Sigma,  # initial covariance
            p_exist,  # current existence probability
            timestamp=new_time
        )])  # birth hypothesis
        survival_track = Track(track)  # use previous track from line 8 or 98
        priors = [birth_track, survival_track]  # store birth_track and surviving track
        mus = StateVectors([state.state_vector for state in priors])  # priori state vectors
        Sigmas = numpy.stack([state.covar for state in priors], axis=2)  # priori state covariances
        weights = numpy.array([
            p_birth*(1-p_exist),  # target absence weight
            (1-p_death)*p_exist  # target existence weight
        ])  # priori weights
        mu_merge, Sigma_merge = gm_reduce_single(
            mus,  # a priori birth and survival means
            Sigmas,  # a priori birth and survival covariances
            weights  # a priori existence probability weights
        )  # merge priori densities
        track_merge = GaussianStateWithExistencePrediction(
            mu_merge,  # a priori merged mean
            Sigma_merge,  # a priori merged covariance
            p_exist,  # a priori existence probability
            timestamp=new_time
        )  # new merged a priori track object
        hypotheses = data_associator.associate(
            [track_merge], 
            detections, 
            new_time
        )
        hypotheses = hypotheses[track_merge]  # append prior track
        posteriors = []  # posteriors
        weights = []  # weights
        for hypothesis in hypotheses:  # begin loop
            if not hypothesis:  # prior
                posteriors.append(hypothesis.prediction)  # append prior
            else:  # candidate detection
                posterior_state = updater.update(hypothesis)  # update
                posteriors.append(posterior_state)  # append measurement posterior
            weights.append(hypothesis.probability)  # append weight
    elif merge_strategy == 'a posteriori':  # method for combining all a posteriori hypotheses
        # state density for birth hypothesis
        birth_track = Track([GaussianStateWithExistence(
            mu,  # initial mean
            Sigma,  # initial covariance
            p_birth*(1-p_exist),  # current existence probability
            timestamp=new_time
        )])  # birth hypothesis
        birth_hypotheses = data_associator.associate(
            [birth_track], 
            detections, 
            new_time
        )
        birth_hypotheses = birth_hypotheses[birth_track]  # append birth track 
        posteriors = []  # posteriors
        weights = []  # weights
        for hypothesis in birth_hypotheses:  # begin loop
            if not hypothesis:  # prior
                posteriors.append(hypothesis.prediction)  # append prior
            else:  # candidate detection
                posterior_state = updater.update(hypothesis)  # update
                posteriors.append(posterior_state)  # append measurement posterior
            weights.append(hypothesis.probability)  # append weight
        
        # state
        survival_track = Track([GaussianStateWithExistence(
            track.state_vector,  # initial mean
            track.covar,  # initial covariance
            (1-p_death)*p_exist,  # current existence probability
            timestamp=new_time
        )])  # birth hypothesis
        # survival hypotheses
        survival_hypotheses = data_associator.associate(
            [survival_track], 
            detections, 
            new_time
        )
        survival_hypotheses = survival_hypotheses[survival_track]  # contains estimate, covariance, time and weight 
        for hypothesis in survival_hypotheses:  # begin loop
            if not hypothesis:  # prior
                posteriors.append(hypothesis.prediction)  # append prior
            else:  # candidate detection
                posterior_state = updater.update(hypothesis)  # update
                posteriors.append(posterior_state)  # append measurement posterior
            weights.append(hypothesis.probability)  # append weight
        hypotheses = [birth_hypotheses, survival_hypotheses]
    else:
        raise ValueError('Invalid hypothesis management strategy. Use "a priori" or "a posteriori" merging.')
    
    # merge all hypotheses, whether or not a priori hypotheses have merge
    norm_weights = numpy.array(weights)/sum(weights)
    mus = StateVectors([state.state_vector for state in posteriors])  # posterior states
    Sigmas = numpy.stack([state.covar for state in posteriors], axis=2)  # posterior covariances
    norm_weights = numpy.asarray(norm_weights)  # posterior weights
    post_mu, post_Sigma = gm_reduce_single(
        mus, 
        Sigmas, 
        norm_weights
    )  # merge posteriors
    p_absent = (1-p_exist)*(  # target exists, but misdetected
        poisson.pmf(  # poisson pdf
            len(weights),  # all detections are assumed clutter
            lmbd*V  # expected number of clutter detections
        )
    )  # null absence probability (includes death and not-born)
    weights.append(p_absent)
    norm_weights = numpy.array(weights)/sum(weights)
    p_exist = 1-norm_weights[len(norm_weights)-1]  # update existence probability
    track = GaussianStateWithExistenceUpdate(
        post_mu, 
        post_Sigma, 
        p_exist, 
        hypotheses, 
        timestamp=new_time
    )  # new track
    tracks.append(track)  # append to tracks
    old_time = new_time  # set old time

# plots
fig = pyplot.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
# Plot states
ax.plot([state.state_vector[0, 0] for state in states],
        [state.state_vector[2, 0] for state in states],
        linestyle="--", 
        marker='o'
        )
for _set in detection_sets:
    # Plot actual detections.
    ax.scatter([measurement.state_vector[0] for measurement in _set if isinstance(measurement, TrueDetection)], 
               [measurement.state_vector[1] for measurement in _set if isinstance(measurement, TrueDetection)], 
               color='b'
               )
    # Plot clutter.
    ax.scatter([clutter.state_vector[0] for clutter in _set if isinstance(clutter, Clutter)], 
               [clutter.state_vector[1] for clutter in _set if isinstance(clutter, Clutter)], 
               color='y', 
               marker='2'
               )
# Plot track
ax.plot([track.state_vector[0, 0] for track in tracks], 
        [track.state_vector[2, 0] for track in tracks], 
        linestyle="--", 
        marker='o'
        )
for track in tracks[1:]:   # Skip the prior
    w, v = numpy.linalg.eig(
        measurement_model.matrix()@track.covar@measurement_model.matrix().T
    )
    max_ind = numpy.argmax(w)
    min_ind = numpy.argmin(w)
    orient = numpy.arctan2(v[1, max_ind], v[0, max_ind])
    ellipse = Ellipse(xy=track.state_vector[(0, 2), 0], 
                      width=2*numpy.sqrt(w[max_ind]), 
                      height=2*numpy.sqrt(w[min_ind]), 
                      angle=numpy.rad2deg(orient), 
                      alpha=0.1
                      )
    ax.add_artist(ellipse)

fig.show()

#######################################################################################
# Figure explanation
#
# The figure below presents the target state, measurements, and clutter detections
# throughout the simulation. All detections, including target measurements and clutter, 
# are simulated within the surveillance region bounded by the circular volume V
# determined by the maximum surveillance range rmax.
# The initial target position is set such that the target may appear anywhere within
# the surveillance region; however, although target measurements are bounded by this
# region, the target state is not.


# In[13]:


fig, ax = pyplot.subplots(
    figsize=(10, 6)
)
ax.set_xlabel("$Time$")
ax.set_ylabel("$p(E_{k}|Z_{k})$")
# Plot existence
ax.plot(numpy.arange(0, T+dt, dt), 
        [track.p_exist for track in tracks], 
        color='k', 
        linestyle="--", 
        marker='o'
        )
fig.show()
#######################################################################################
# Figure explanation
# 
# The figure below presents the existence probability at a given time. When the existence
# probability is low, this indicates the target absence probability, as modelled by the
# stochastic birth, death, and clutter quantity models, is greater than the target
# existence probability.
# A common reason for a decrease in the existence probability is the drop in the
# target measurement likelihood probability, which is a symptom of misdetection. This may
# be checked by assessing the available detection data at the time of interest.
# Another reason for a drop in the existence probability occurs when targets leave the
# surveillance region as described in the previous figure. We see that targets that leave
# this region do not generate measurements, but continue to exist outside this region. As
# a result, it is perfectly normal, and expected, that the existence probability will
# decrease when this occurs despite the presence of the target outside the surveillance
# region.

stop = 1  # breakpoint to see visuals before termination
