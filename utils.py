import os
import errno
import datetime
import numpy as np
import logging

from typing import List

import tensorflow as tf


def summary_path(now: str, exp: int = None, env: int = None, name: str = None, filename: str = None):
    """ get_summary_path: returns a file path for collecting summary for each experiment """
    path_list = [os.path.dirname(__file__), "summary", now]
    if exp is not None:
        path_list.append(f"Exp{exp:02}")
    if env is not None:
        path_list.append(f"Env{env:02}")
    if name is not None:
        path_list.append(name)
    if filename is not None:
        path_list.append(filename)
    file_path = os.path.join(*path_list)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return file_path


def variable_summary(summary_writer, scope_name: str, variable_name: str, step: int, value):
    if summary_writer:
        with summary_writer.as_default(step=step), tf.name_scope(scope_name):
            if isinstance(value, list) and len(value) > 0:
                with tf.name_scope(variable_name):
                    values = [float(v) for v in value]
                    tf.summary.scalar('mean', np.mean(values))
                    tf.summary.scalar('max', np.max(values))
                    tf.summary.scalar('min', np.min(values))
                    tf.summary.scalar('sum', np.sum(values))
                    tf.summary.scalar('stddev', np.std(values))
            elif isinstance(value, (int, float)):
                tf.summary.scalar(variable_name, value)


def info(text):
    t = f"[{datetime.datetime.now()}] {str(text)}"
    logging.info(t)
    print(t)


def debug(text):
    logging.debug(f"[{datetime.datetime.now()}] {str(text)}")


def compute_value(rewards: List[float], discount_factor: float) -> List[float]:
    values = [rewards[-1]]
    for i in reversed(range(len(rewards) - 1)):
        values.append(rewards[i] + discount_factor * values[-1])
    values.reverse()
    assert len(rewards) == len(values)
    return values


def save_gif(frames, path: str):
    assert ".gif" in path
    if len(frames) > 0:
        frames[0].save(path, append_images=frames[1:], format='GIF', save_all=True, duration=1000)


def clamp(value: float, lower: float, upper: float):
    assert lower < upper
    return min(upper, max(lower, value))


def calculate_score(lower: float, higher: float, value: float) -> float:
    assert lower <= higher
    if lower == higher:
        return 1.0 if value >= higher else 0.0
    return (value - lower) / (higher - lower)


def get_distances(coordinates):
    distances = []
    for a in coordinates:
        for b in coordinates:
            if a != b:
                distances.append(a.distance(b))
    return distances


class ExperienceMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity

        self.observation = []
        self.service_state = []
        self.context = []
        self.reward = []

    def add(self, observation: List[float], service_state: List[float], context: List[float], reward: float):
        while self.capacity and self.length() > self.capacity:
            self.pop()
        self.observation.append(observation)
        self.service_state.append(service_state)
        self.context.append(context)
        self.reward.append(reward)

    def sample(self):
        sample = {
            "observation": np.array(self.observation),
            "service_state": np.array(self.service_state),
            "context": np.array(self.context),
            "reward": np.array(self.reward),
        }
        if not self.capacity:
            self.observation.clear()
            self.service_state.clear()
            self.context.clear()
            self.reward.clear()
        return sample

    def length(self) -> int:
        assert len(self.observation) == len(self.service_state) == len(self.context) == len(self.reward)
        return len(self.observation)

    def is_full(self) -> bool:
        return self.length() >= self.capacity

    def pop(self):
        return {
            "observation": self.observation.pop(0),
            "service_state": self.service_state.pop(0),
            "context": self.context.pop(0),
            "reward": self.reward.pop(0),
        }


class Vector:
    """ Vector: class of 3-dimensional vector for Coordinate and Direction """
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def update(self, x: float, y: float, z: float):
        """ update: updates the elements of the vector """
        self.x = x
        self.y = y
        self.z = z

    def vectorize(self):
        """ vectorize: returns list form of the vector, for concatenation with other lists """
        return [self.x, self.y, self.z]

    def tuple(self):
        return self.x, self.y, self.z

    def size(self) -> float:
        """ size: returns the size of the vector """
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def dot(self, other) -> float:
        """ dot: performs dot product of vectors """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """ cross: performs cross product of vectors """
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    def get_cosine_angle(self, other):
        """ get_cosine_angle: calculates cosine value between the vector and target vector """
        assert isinstance(other, Vector)
        return round(self.dot(other) / (self.size() * other.size()), 3)

    def get_angle(self, other):
        """ get_angle: calculates angle between the vector and target vector, in degree scale """
        return np.degrees(np.arccos(self.get_cosine_angle(other)))

    def distance(self, other) -> float:
        """ get_distance: get distance between to vectors """
        assert isinstance(other, Vector)
        return (other - self).size()

    def horizontal_distance(self, other) -> float:
        return Vector(other.x - self.x, other.y - self.y, 0).size()

    def unit(self):
        """ to_unit: converts the vector to a unit vector that is parallel to the vector but size 1 """
        denominator = self.size()
        u = Vector(x=self.x/denominator, y=self.y/denominator, z=self.z/denominator)
        assert round(u.size(), 3) == 1
        return u

    def between(self, a, b) -> bool:
        """ between: determine whether the vector is between a and b """
        return (self.x - a.x) * (self.x - b.x) <= 0 and (self.y - a.y) * (self.y - b.y) <= 0

    def copy(self):
        return Vector(self.x, self.y, self.z)

    def round(self, digit=0):
        return Vector(round(self.x, digit), round(self.y, digit), round(self.z, digit))

    def __str__(self):
        return "(X:{x}, Y:{y}, Z:{z})".format(x=self.x, y=self.y, z=self.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector(self.x + other, self.y + other, self.z + other)
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector(other + self.x, other + self.y, other + self.z)
        return Vector(other.x + self.x, other.y + self.y, other.z + self.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __rsub__(self, other):
        return Vector(other.x - self.x, other.y - self.y, other.z - self.z)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return Vector(other * self.x, other * self.y, other * self.z)

    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
