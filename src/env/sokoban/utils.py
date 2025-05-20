import random
from contextlib import contextmanager
import numpy as np
from gym import logger

@contextmanager
def set_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)

@contextmanager
def NoLoggerWarnings():
    logger.set_level(logger.ERROR)
    try:
        yield
    finally:
        logger.set_level(logger.INFO)