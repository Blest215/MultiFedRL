import numpy as np

from PIL import Image
from typing import List, Optional

from utils import Vector, clamp


class Coordinate(Vector):
    def __init__(self, x: float, y: float, z: float, momentum=None):
        super().__init__(x, y, z)
        assert momentum is None or isinstance(momentum, Mobility)
        self.momentum = momentum if momentum else Mobility.random()

    def copy(self):
        return Coordinate(self.x, self.y, self.z, self.momentum)

    def move(self, env, new_coordinate):
        assert isinstance(new_coordinate, Coordinate)

        while env.block(self, new_coordinate):
            distance = abs(np.random.normal(0, (new_coordinate - self).size()))
            angle = np.random.uniform(0, 2 * np.pi)
            new_coordinate = Coordinate(
                x=self.x + distance * np.cos(angle), y=self.y + distance * np.sin(angle), z=self.z
            )
            self.momentum = Mobility(distance, angle)

        self.update(
            x=clamp(new_coordinate.x, 0, env.width),
            y=clamp(new_coordinate.y, 0, env.height),
            z=clamp(new_coordinate.z, 0, env.depth)
        )

    def random_walk(self, env, distance: float):
        angle = np.random.uniform(0, 2 * np.pi)
        new_coordinate = Coordinate(
            x=self.x + distance * np.cos(angle), y=self.y + distance * np.sin(angle), z=self.z
        )
        self.move(env, new_coordinate)

    def move_toward(self, env, destination, max_distance: float):
        assert isinstance(destination, Coordinate)
        distance = np.random.uniform(0, max_distance)
        if self.horizontal_distance(destination) <= distance:
            new_coordinate = Coordinate(x=destination.x, y=destination.y, z=self.z)
        else:
            angle = np.arctan((destination.y - self.y) / (destination.x - self.x))
            new_coordinate = Coordinate(
                x=self.x + distance * np.cos(angle), y=self.y + distance * np.sin(angle), z=self.z
            )
        self.move(env, new_coordinate)

    def move_pattern(self, env, speed_scale: float, momentum_ratio: float):
        patterns = [
            env.mobility_map[x][y]
            for x, y in [
                (int(np.trunc(self.x)), int(np.trunc(self.y))),
                (int(np.trunc(self.x)), int(np.ceil(self.y))),
                (int(np.ceil(self.x)), int(np.trunc(self.y))),
                (int(np.ceil(self.x)), int(np.ceil(self.y)))
            ]
            if not env.block(Coordinate(self.x, self.y, 0), Coordinate(x, y, 0))
        ]

        distance = abs(np.random.normal(0, speed_scale))

        if patterns:
            mobility = sum(patterns) / len(patterns)

            if mobility.is_opposite(self.momentum):
                mobility = -mobility

            self.momentum = self.momentum * momentum_ratio + mobility * (1 - momentum_ratio)

        self.move(env, self.momentum.sample(self, distance))


class Mobility:
    def __init__(self, radius: float, theta: float):
        assert 0 <= radius
        self.radius = radius
        self.theta = theta % (2 * np.pi)

    @staticmethod
    def random():
        return Mobility(np.random.uniform(0.5, 1), np.random.uniform(0, 2 * np.pi))

    @property
    def x(self):
        return self.radius * np.cos(self.theta)

    @property
    def y(self):
        return self.radius * np.sin(self.theta)

    def is_opposite(self, mobility) -> bool:
        assert isinstance(mobility, Mobility)
        return np.pi / 2 <= abs(self.theta - mobility.theta) < 3 * np.pi / 2

    def sample(self, coordinate: Coordinate, distance: float) -> Coordinate:
        radius = abs(np.random.normal(0, self.radius))
        theta = np.random.normal(self.theta)
        return Coordinate(
            x=coordinate.x + radius * np.cos(theta) * distance,
            y=coordinate.y + radius * np.sin(theta) * distance,
            z=coordinate.z
        )

    def __neg__(self):
        return Mobility(self.radius, self.theta + np.pi)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Mobility(self.radius + other, self.theta)
        return Mobility(self.radius + other.radius, self.theta + other.theta)

    def __radd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Mobility(other + self.radius, self.theta)
        return Mobility(other.radius + self.radius, other.theta + self.theta)

    def __sub__(self, other):
        return Mobility(self.radius - other.radius, self.theta - other.theta)

    def __rsub__(self, other):
        return Mobility(other.radius - self.radius, other.theta - self.theta)

    def __mul__(self, other):
        return Mobility(self.radius * other, self.theta * other)

    def __rmul__(self, other):
        return Mobility(other * self.radius, other * self.theta)

    def __truediv__(self, other):
        return Mobility(self.radius / other, self.theta / other)

    def __eq__(self, other):
        return isinstance(other, Mobility) and self.radius == other.radius and self.theta == other.theta


class Direction(Vector):
    """ Direction: class that represents direction of a physical entity in a 3-dimensional space """
    def __init__(self, x: Optional[float], y: Optional[float], z: Optional[float]):
        """ Direction should be a unit vector or zero vector """
        if x == 0 and y == 0 and z == 0:
            Vector.__init__(self, x=x, y=y, z=z)
        else:
            # Randomize if the value is None
            if x is None:
                x = np.random.uniform(-1, 1)
            if y is None:
                y = np.random.uniform(-1, 1)
            if z is None:
                z = np.random.uniform(-1, 1)
            denominator = np.sqrt(np.square(x) + np.square(y) + np.square(z))
            Vector.__init__(self, x=x / denominator, y=y / denominator, z=z / denominator)
        assert np.isclose(self.size(), 1) or np.isclose(self.size(), 0)

    @staticmethod
    def from_vector(vector):
        return Direction(vector.x, vector.y, vector.z)

    def rotate_xy(self, radian):
        """ rotate_xy: rotate the vector along the xy-plane, ignores z """
        cos = np.cos(radian)
        sin = np.sin(radian)

        self.x = self.x * cos - self.y * sin
        self.y = self.x * sin + self.y * cos


class Body:
    """ Physical body """
    def __init__(self, coordinate: Coordinate, orientation: Direction):
        self.coordinate = coordinate
        self.orientation = orientation
        self.icon = self.load_icon()

    def distance(self, other) -> float:
        assert isinstance(other, Body)
        return self.coordinate.distance(other.coordinate)

    def load_icon(self) -> Image:
        pass


class Service(Body):
    def __init__(self, coordinate: Coordinate, orientation: Direction, sid: int):
        super().__init__(coordinate, orientation)

        self.sid = sid
        self.agent = None

        # State
        self.user = None
        self.duration = 0
        self.count = 0  # Number of provision, count after release

    def __str__(self):
        return f"{self.__class__.__name__} {self.sid}"

    def prediction(self, sample):
        assert not any(self.state())
        return self.agent.prediction(samples={
            "observation": np.array([sample["observation"]]),
            "service_state": np.array([self.state()]),
            "context": np.array([sample["context"]["communication" if self.agent.enable_communication else "raw"]]),
        }, training=False).numpy().item()

    def communication(self, sample):
        assert any(self.state())
        return np.squeeze(self.agent.communication(samples={
            "observation": np.array([sample["observation"]]),
            "service_state": np.array([self.state()]),
            "context": np.array([sample["context"]["communication"]]),
        }, training=False).numpy())

    def acquire(self, user, duration: int):
        self.user = user
        self.duration = duration
        self._acquire(user, duration)

    # Type-specific acquire function
    def _acquire(self, user, duration: int):
        pass

    def release(self):
        self.user = None
        self.duration = 0
        self.count += 1
        self._release()

    # Type-specific release function
    def _release(self):
        pass

    def reset(self):
        assert not self.user
        self.duration = 0
        self.count = 0
        self._reset()

    def adjust(self, env):
        count = 100
        while (env.block(
            self.coordinate, self.coordinate + self.orientation
        ) or env.outside(
            self.coordinate + self.orientation
        )) and count > 0:
            self.orientation = Direction(x=None, y=None, z=0)
            count -= 1

    # Type-specific reset function
    def _reset(self):
        pass

    # Type-specific effectiveness calculation
    def effectiveness(self, env, user) -> bool:
        pass

    # Type-specific control behavior of users
    def control(self, env) -> bool:
        pass

    # Type-specific state
    def state(self) -> List[float]:
        pass

    def freeze(self) -> dict:
        return {
            "coordinate": self.coordinate.copy(),
        }

    def resume(self, history: dict):
        self.coordinate = history["coordinate"].copy()

    # Agent assignment
    def set(self, env, agent):
        self.agent = agent
        self.agent.set(env, self)

    def remove_agent(self):
        del self.agent.model
        del self.agent.communication_model
        del self.agent.memory
        del self.agent
        self.agent = None


class SpeakerService(Service):
    def __init__(self, coordinate: Coordinate, orientation: Direction, sid: int, configuration):
        super().__init__(coordinate, orientation, sid)

        # Pattern
        self.intensity_range = [0, np.random.randint(*configuration.speaker_maximum_intensity)]
        self.adjust_step = configuration.speaker_adjust_step

        # State
        self.intensity = 0  # 1m reference

    def load_icon(self):
        return Image.open('rendering/speaker.png', 'r').convert("RGBA")

    def set_intensity(self, intensity: int):
        self.intensity = clamp(value=intensity, lower=self.intensity_range[0], upper=self.intensity_range[1])

    def _release(self):
        self.set_intensity(0)

    def _reset(self):
        self.set_intensity(0)

    def effectiveness(self, env, user):
        intensity = self.perceived_intensity(env, user)
        return intensity >= self.interference(env, user) and intensity >= user.acoustic_acuity

    def interference(self, env, user):
        sounds = [env.background_noise] + [
            service.perceived_intensity(env, user)
            for service in env.services
            if isinstance(service, SpeakerService) and service is not self and service.intensity > 0
        ]
        return 10 * np.log10(sum([pow(10, n / 10) for n in sounds if n > 0]))

    def control(self, env):
        intensity = self.intensity
        while self.effectiveness(env, self.user) and self.intensity > self.intensity_range[0]:
            self.set_intensity(self.intensity - self.adjust_step)

        while not self.effectiveness(env, self.user) and self.intensity < self.intensity_range[1]:
            self.set_intensity(self.intensity + self.adjust_step)
        return intensity != self.intensity

    def perceived_intensity(self, env, user) -> float:
        if self.intensity <= 0:
            return 0

        intensity = self.intensity - 20 * np.log10(user.distance(self))

        for wall, intersection in env.block(self.coordinate, user.coordinate):
            intensity += 20 * np.log10(1 - wall.absorption_rate)

        return intensity

    def state(self):
        return [self.intensity]


class DisplayService(Service):
    def __init__(self, coordinate: Coordinate, orientation: Direction, sid: int, configuration):
        super().__init__(coordinate, orientation, sid)

        # Pattern
        self.text_size_range = [0, np.random.randint(*configuration.display_maximum_text_size)]
        self.adjust_step = configuration.display_adjust_step

        # State
        self.text_size = 0

    def load_icon(self):
        return Image.open('rendering/display.png', 'r').convert("RGBA")

    def set_text_size(self, text_size: int):
        self.text_size = clamp(value=text_size, lower=self.text_size_range[0], upper=self.text_size_range[1])

    def _release(self):
        self.set_text_size(0)

    def _reset(self):
        self.set_text_size(0)

    def effectiveness(self, env, user):
        # Blocking rule
        # There should be no walls between the user and the device
        if env.block(self.coordinate, user.coordinate):
            return False

        # Visual field rule
        # The device should be inside of the user's visual field
        relative_location = self.coordinate - user.coordinate
        theta = relative_location.get_angle(user.orientation)
        if theta > user.visual_field_max:
            return False

        # Orientation rule
        # The face of the visual display should be opposite of the user's face
        psi = self.orientation.get_angle(-relative_location)
        if psi > user.viewing_angle_max:
            return False

        # Visual angle rule
        # 6/6 vision is defined as: at 6 m distance, human can recognize 5 arc-min letter.
        # so size of the minimum letter is: 2 * 6 * tan(5 / 120) = 0.00873 m
        # Converting text size in point to meter: 1 point is 0.000352778 meter
        text_size = self.text_size * 0.000352778
        perceived_size = text_size * self.orientation.get_cosine_angle(-relative_location)
        visual_angle = np.degrees(2 * np.arctan(perceived_size / (2 * user.distance(self))))
        if visual_angle / 5 < user.minimum_visual_angle:
            return False

        return True

    def control(self, env):
        text_size = self.text_size
        # Rotate head
        self.user.orientation = Direction.from_vector(self.coordinate - self.user.coordinate)

        # Increase text size
        while not self.effectiveness(env, self.user) and self.text_size < self.text_size_range[1]:
            self.set_text_size(self.text_size + self.adjust_step)
        return text_size != self.text_size

    def state(self):
        return [self.text_size]


class CoolerService(Service):
    def __init__(self, coordinate: Coordinate, orientation: Direction, sid: int, configuration):
        super().__init__(coordinate, orientation, sid)

        # Pattern
        self.maximum_range = configuration.cooler_maximum_range
        self.maximum_distance = np.random.randint(*configuration.cooler_maximum_distance)
        self.adjust_step = np.random.randint(*configuration.cooler_maximum_adjust_step)

        # State
        self.temperature = 0

        self.x = round(self.coordinate.x)
        self.y = round(self.coordinate.y)

        self.target = None

    def load_icon(self):
        return Image.open('rendering/cooler.png', 'r').convert("RGBA")

    def _acquire(self, user, duration):
        assert user.temperature_expected is not None
        self.temperature = user.temperature_expected

    def _release(self):
        self.temperature = 0

    def _reset(self):
        self.temperature = 0

    def effectiveness(self, env, user):
        return abs(
            env.temperature[round(user.coordinate.x)][round(user.coordinate.y)]
            - user.temperature_expected
        ) <= user.temperature_buffer

    def control(self, env):
        def spread(m, n, depth):
            if depth < self.maximum_range:
                if (m, n) not in self.target:
                    self.target.append((m, n))
                for i, j in env.adjacency_map[m][n]:
                    spread(i, j, depth + 1)

        # Calculate target temperature
        update_temperature = env.temperature[self.x][self.y] + clamp(
            self.temperature - env.temperature[self.x][self.y],
            -self.adjust_step, self.adjust_step
        )
        changed = update_temperature != env.temperature[self.x][self.y]

        # Get targets
        if not self.target:
            self.target = []
            for distance in range(self.maximum_distance):
                target = Coordinate(
                    x=round(self.coordinate.x + self.orientation.x * distance),
                    y=round(self.coordinate.y + self.orientation.y * distance),
                    z=self.coordinate.z
                )
                if not env.block(self.coordinate, target) and not env.outside(target):
                    spread(target.x, target.y, 0)

        # Update temperature
        for x, y in self.target:
            env.temperature[x][y] = update_temperature
            env.temperature_fix[x][y] = True

        return changed

    def state(self):
        return [self.temperature]


class User(Body):
    def __init__(self, coordinate: Coordinate, orientation: Direction, uid: int, configuration):
        super().__init__(coordinate, orientation)
        assert configuration.user_height_range[0] <= coordinate.z <= configuration.user_height_range[1]

        self.uid = uid

        # Action
        self.enter_probability = configuration.user_enter_probability
        self.request_probability = configuration.user_request_probability
        self.exit_probability = configuration.user_exit_probability
        self.duration_range = configuration.user_duration_range
        self.speed_scale = configuration.user_speed_scale
        self.momentum_ratio = configuration.momentum_ratio
        self.service_types = list(configuration.num_services.keys())
        self.exploration = configuration.exploration

        # Property
        self.visual_acuity = configuration.user_visual_acuity
        self.minimum_visual_angle = pow(10, self.visual_acuity) / 60
        self.visual_field_max = configuration.user_visual_field_max
        self.viewing_angle_max = configuration.user_viewing_angle_max
        self.acoustic_acuity = configuration.user_acoustic_acuity
        self.temperature_range = configuration.user_temperature_expected
        self.temperature_expected = None
        self.temperature_buffer = configuration.user_temperature_buffer

        # Feedback
        self.feedback_probability = configuration.user_feedback_probability
        self.feedback_dense = configuration.user_feedback_dense
        self.effectiveness_scale = configuration.user_effectiveness_scale

        # State
        self.active = False
        self.request = None
        self.service = None
        self.episode = None

    def reset(self):
        assert not self.service
        self.active = False
        self.request = None
        self.service = None
        self.episode = None

    def load_icon(self):
        return Image.open('rendering/user.png', 'r').convert("RGBA")

    def selection(self, candidates: List[Service], observation, context, train: bool) -> Service:
        Q = [service.prediction({"observation": observation, "context": context}) for service in candidates]
        assert len(Q) == len(candidates)

        # Exploration
        if self.exploration and train and all([q <= 0 for q in Q]):
            return np.random.choice(candidates)

        # Greedy
        return candidates[np.argmax(Q)]

    def step(self, env):
        if not self.active and np.random.random() < self.enter_probability:
            self.coordinate = env.get_random_user_coordinate()
            self.temperature_expected = np.random.randint(*self.temperature_range)
            self.active = True
        elif self.active and np.random.random() < self.exit_probability and not self.service:
            self.active = False

        # Mobility pattern
        self.coordinate.move_pattern(env, self.speed_scale, self.momentum_ratio)
        if env.outside(self.coordinate):
            self.active = False

        # Request pattern
        if self.active and not self.service and not self.request and np.random.random() < self.request_probability:
            self.request = self.new_request()

    def state(self) -> List[float]:
        return self.coordinate.vectorize()

    def feedback(self, env) -> float:
        assert self.service and self.episode
        if np.random.random() > self.feedback_probability:
            return 0
        return self.effectiveness_scale if self.service.effectiveness(env, self) else -self.effectiveness_scale

    def new_request(self):
        return Request(
            service=np.random.choice(self.service_types),
            duration=np.random.randint(low=self.duration_range[0], high=self.duration_range[1])
            if isinstance(self.duration_range, list) else self.duration_range,
        )

    def remove_request(self):
        self.request = None

    def acquire(self, service: Service):
        assert self.request and not self.episode and not self.service and not service.user
        service.acquire(self, self.request.duration)
        self.service = service
        self.episode = Episode(self, self.request, service)

    def release(self):
        assert self.service and self.episode
        self.service.release()
        self.service = None

        episode = self.episode
        self.episode = None
        return episode

    def freeze(self) -> dict:
        return {
            "active": self.active,
            "coordinate": self.coordinate.copy(),
            "request": self.request.copy() if self.request else None,
        }

    def resume(self, history: dict):
        self.active = history["active"]
        self.coordinate = history["coordinate"].copy()
        self.request = history["request"].copy() if history["request"] else None


class Wall:
    def __init__(self, a: Vector, b: Vector, configuration):
        # wall is from a-endpoint to b-endpoint, and orthogonal to ground
        self.a = a
        self.b = b

        sub = (b - a)
        self.normal = Vector(x=1, y=(-sub.x / sub.y), z=0) if sub.y != 0 else Vector(x=(-sub.y / sub.x), y=1, z=0)

        self.absorption_rate = np.random.uniform(*configuration.wall_sound_absorption_rate_range)

    def intersect(self, start: Vector, end: Vector):
        direction = end - start
        denominator = self.normal.dot(direction)

        # Parallel, ignore contained case
        if denominator == 0:
            return None

        # Intersect
        intersection = start + (self.normal.dot(self.a - start) / denominator) * direction

        if intersection.between(start, end) and intersection.between(self.a, self.b):
            return intersection
        return None


class Request:
    def __init__(self, service: Service, duration: int):
        self.service = service
        self.duration = duration

    def copy(self):
        return Request(self.service, self.duration)


class Episode:
    def __init__(self, requester: User, request: Request, service: Service):
        assert requester and request and service
        self.requester = requester
        self.request = request
        self.service = service

        self.total_duration = request.duration

        self.observations = []
        self.service_states = []
        self.contexts = []
        self.rewards = []

    def add_observation(self, observation: List[float], context, reward):
        self.observations.append(observation)
        self.service_states.append(self.service.state())
        self.contexts.append(context)
        self.rewards.append(reward)

    def summary(self):
        assert len(self.observations) == len(self.service_states) == len(self.contexts) == len(self.rewards)
        assert len(self.rewards) <= self.total_duration

        self.service.agent.add([{
            "observation": self.observations[i],
            "service_state": self.service_states[i],
            "context": self.contexts[i],
            "reward": sum(self.rewards[i:]) / len(self.rewards[i:]) if self.requester.feedback_dense
            else sum(self.rewards) / len(self.rewards),
        } for i in range(len(self.observations))])
