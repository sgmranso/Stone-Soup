# -*- coding: utf-8 -*-
import numpy as np
from operator import attrgetter

from .base import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle


class MultinomialResampler(Resampler):

    def resample(self, particles):
        """
        Resample the particles using multinomial resampling.
        :param particles: list of :class:`~.Particle`
            The particles to be resampled according to their weight
        :return particles : list of :class:`~.Particle`
            The resampled particles
        """
        weights = [particle.weight for particle in particles]  # particle weights
        bins = np.cumsum(weights)  # bins based on weights
        bins = np.array(bins, dtype='float64')  # format for digitize
        bins = np.insert(bins, 0, 0)  # insert zero
        bins[len(particles)] = 1  # exact upper edge
        rands = np.random.rand(len(particles))  # uniform samples for bins
        indices = np.digitize(rands, bins)  # assign samples to bin indices
        choose_particles = []  # allocate array for new particles
        for index in indices:  # loop over indices
            choose_particles.append(Particle(  # append new particle
                    particles[index].state_vector,  # new particle state vector
                    weight=Probability(1/len(particles)),  # assume all particle weights are uniform
                    parent=particles[index]  # new particle
            ))

        new_particles = choose_particles
        # choose_weights = [p.weight for p in choose_particles]  # particle weights
        # choose_weights = np.array(choose_weights, dtype='float64')  # format
        # new_weights = choose_weights/sum(choose_weights)  #
        # new_particles = []  # allocate array for new particles
        # for index in range(len(particles)):  # loop over particles
        #     new_particles.append(Particle(  # append new particle
        #             choose_particles[index].state_vector,  # new particle state vector
        #             weight=Probability(new_weights[index]),  # assume all particle weights are uniform
        #             parent=choose_particles[index]  # new particle
        #     ))

        return new_particles


class SystematicResampler(Resampler):

    def resample(self, particles):
        """Resample the particles

        Parameters
        ----------
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.Particle`
            The resampled particles
        """

        n_particles = len(particles)
        weight = Probability(1/n_particles)
        particles_sorted = sorted(particles, key=attrgetter('weight'), reverse=False)
        cdf = np.cumsum([p.weight for p in particles_sorted])

        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)
        new_particles = []

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        for j in range(n_particles):

            u_j = u_i + (1 / n_particles) * j

            particle = particles_sorted[np.argmax(u_j < cdf)]
            new_particles.append(
                Particle(particle.state_vector,
                         weight=weight,
                         parent=particle))

        return new_particles
