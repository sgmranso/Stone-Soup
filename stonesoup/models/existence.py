# -*- coding: utf-8 -*-
import math
from typing import Sequence
from functools import lru_cache

import numpy as np
from .base import ExistenceModel
from ..base import Property
from ..types.numeric import Probability
from ..types.state import GaussianState


class BirthModel(ExistenceModel):
    """
    A birth model class.
    """

    birth_probability: Probability = Property(doc="Birth model for a new track.")

    birth_density: GaussianState = Property(doc='The birth density', default=None)

    def birth_model(self):
        birth_probability = Probability(self.model_probability)
        return birth_probability

class DeathModel(ExistenceModel):
    """
    A death model class.
    """

    death_probability: Probability = Property(doc="Death model for an existing track.")
    
    def death_model(self):
        death_probability = Probability(self.model_probability)
        return death_probability