#!/usr/bin/env python
# coding: utf-8

# Some general imports and set up
import numpy
from matplotlib import pyplot
from matplotlib.patches import Ellipse
from datetime import timedelta
from datetime import datetime
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from scipy.stats import multivariate_normal
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.types.state import ParticleState
from stonesoup.types.particle import Particle
from stonesoup.types.numeric import Probability
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track


# simulation variables
start_time = datetime.now()
T = 20  # temporal length
dt = 1  # time step length
dim = 2  # position dimensions
sigma_b = numpy.array([100, 5])  # birth covariance coefficient
sigma_x = 3  # process noise coefficient, q
sigma_z = 3  # measurement noise coefficient, r
Ns = 1000  # number of particles
# birth density
mu = numpy.array(numpy.repeat([0], 4))  # zero-mean state
Sigma = numpy.diag(numpy.tile(numpy.power(sigma_b, 2), 2))  # birth density
# transition model
transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(numpy.power(sigma_x, 2)), 
     ConstantVelocity(numpy.power(sigma_x, 2)))
)
# measurement model
measurement_model = LinearGaussian(
    4,  # Number of state dimensions (position and velocity in 2D)
    (0, 2),  # Mapping measurement vector index to state index
    numpy.diag(numpy.repeat(numpy.power([sigma_z], 2), 2))
)

# simulate groundtruth
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
        ), 
        timestamp=new_time
    )
    states.append(state)
    old_time = new_time

# simulate target detections
measurements = []
for state in states:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(
        state_vector=measurement,
        timestamp=state.timestamp
    ))
# particle filter objects
resampler = SystematicResampler()
predictor = ParticlePredictor(transition_model)
updater = ParticleUpdater(measurement_model, resampler)
samples = multivariate_normal.rvs(mu, Sigma, size=Ns)
particles = [Particle(sample.reshape(-1, 1), weight=Probability(1/Ns)) for sample in samples]
track = ParticleState(particles, timestamp=start_time)
# tracker
tracks = Track([track])
for measurement in measurements:
    prediction = predictor.predict(track, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Used to group a prediction and measurement together
    track = updater.update(hypothesis)
    tracks.append(track)
    track = tracks[-1]

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
# Plot detections
ax.scatter([measurement.state_vector[0] for measurement in measurements],
           [measurement.state_vector[1] for measurement in measurements],
           color='b'
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

for state in tracks:
    data = numpy.array([particle.state_vector for particle in state.particles])
    ax.plot(data[:, 0], data[:, 2], linestyle='', marker=".", markersize=1, alpha=0.5)

fig.show()
stop = 1  # breakpoint for visuals prior to termination
