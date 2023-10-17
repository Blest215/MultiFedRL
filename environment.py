import numpy as np

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict

from settings import record_gif
from models import User, Service, Wall, Coordinate, Mobility, Direction
from utils import clamp, get_distances, Vector
from agent import ServiceAgent


class Environment:
    def __init__(self, index: int, configuration):
        self.index = index
        self.configuration = configuration

        # Settings
        self.width = configuration.width
        self.height = configuration.height
        self.depth = configuration.depth

        self.wall_length_ratio = configuration.wall_length_ratio
        self.minimum_service_average_distance_ratio = configuration.minimum_service_average_distance_ratio
        self.minimum_service_distance = configuration.minimum_service_distance
        self.temperature_propagation_iterations = configuration.temperature_propagation_iterations
        self.block_precision_level = configuration.block_precision_level

        # States
        self.channel = np.zeros(configuration.embedding_size).tolist()
        self.background_noise = configuration.background_noise
        self.background_temperature = configuration.background_temperature
        self.temperature = np.full((self.height + 1, self.width + 1), self.background_temperature, dtype=float)
        self.temperature_fix = np.full((self.height + 1, self.width + 1), False)

        # Block approximation cache
        self.block_cache = {}

        # Entities
        self.users = []
        self.set_users(configuration.num_users)
        self.service_types = list(configuration.num_services.keys())
        self.services = []
        self.set_services(configuration.num_services)
        self.walls = []
        self.set_walls(configuration)

        for service in self.services:
            service.adjust(self)

        # Mobility pattern
        self.mobility_map = None
        self.set_mobility_patterns()

        # Adjacency of each coordinate
        self.adjacency_map = None
        self.set_adjacency_map()

        # Trace
        self.training_history = {}
        self.testing_history = {}

        # Rendering
        self.scale = 50
        self.font = ImageFont.truetype("arial.ttf", int(self.scale / 2)) if record_gif else None
        self.small_font = ImageFont.truetype("arial.ttf", int(self.scale / 4)) if record_gif else None
        self.field = None

    def set_walls(self, configuration):
        assert isinstance(configuration.num_walls, int) or isinstance(configuration.num_walls, list)
        num_walls = configuration.num_walls if isinstance(
            configuration.num_walls, int
        ) else np.random.randint(*configuration.num_walls)

        self.walls = []
        for i in range(num_walls):
            a = self.get_random_coordinate()
            a.x = clamp(int(np.trunc(a.x)) + 0.5, 0, self.width)
            a.y = clamp(int(np.trunc(a.y)) + 0.5, 0, self.height)
            a.z = 0
            direction = np.random.randint(0, 4)
            length = np.random.uniform(*self.wall_length_ratio)
            if direction == 0:  # Up
                length = 1 + np.round(self.width * length)
                b = Coordinate(a.x + length, a.y, self.depth)
            elif direction == 1:  # Down
                length = 1 + np.round(self.width * length)
                b = Coordinate(a.x - length, a.y, self.depth)
            elif direction == 2:  # Right
                length = 1 + np.round(self.height * length)
                b = Coordinate(a.x, a.y + length, self.depth)
            else:  # Left
                length = 1 + np.round(self.height * length)
                b = Coordinate(a.x, a.y - length, self.depth)
            self.walls.append(Wall(a=a, b=b, configuration=configuration))

    def get_service_distribution(self):
        n = {}
        dead_space = {}
        mean_distance = {}
        for service_type in self.service_types:
            n[service_type.__name__] = len([service for service in self.services if isinstance(service, service_type)])
            dead = []
            for x in range(self.width + 1):
                for y in range(self.height + 1):
                    location = Coordinate(x, y, self.depth / 2)
                    if all([
                        self.block(service.coordinate, location)
                        for service in self.services if isinstance(service, service_type)
                    ]):
                        dead.append(location)
            dead_space[service_type.__name__] = round(len(dead) / ((self.width + 1) * (self.height + 1)), 3)
            mean_distance[service_type.__name__] = round(np.mean(get_distances([
                service.coordinate for service in self.services if isinstance(service, service_type)
            ])) / self.width, 3) if n[service_type.__name__] > 1 else 1.0
        return n, dead_space, mean_distance

    def set_mobility_patterns(self):
        self.mobility_map = []
        for x in range(self.width + 1):
            line = []
            for y in range(self.height + 1):
                mobility = Mobility.random()
                while self.block(
                    Coordinate(x - mobility.x, y - mobility.y, 0),
                    Coordinate(x + mobility.x, y + mobility.y, 0)
                ):
                    mobility = Mobility.random()
                line.append(mobility)
            self.mobility_map.append(line)

    def set_adjacency_map(self):
        self.adjacency_map = [[[] for _ in range(self.height + 1)] for _ in range(self.width + 1)]
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                for i, j in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    if 0 <= x + i <= self.width and 0 <= y + j <= self.height and not self.block(
                            Coordinate(x, y, self.depth / 2), Coordinate(x + i, y + j, self.depth / 2)
                    ):
                        self.adjacency_map[x][y].append((x + i, y + j))

    def propagate_temperature(self):
        for _ in range(self.temperature_propagation_iterations):
            new_temperature = np.zeros_like(self.temperature)
            for x in range(self.width + 1):
                for y in range(self.height + 1):
                    vicinity = [self.background_temperature] + [
                        self.temperature[i][j] for i, j in self.adjacency_map[x][y]
                    ]
                    new_temperature[x][y] = sum(vicinity) / len(vicinity)

            for x in range(self.width + 1):
                for y in range(self.height + 1):
                    if not self.temperature_fix[x][y]:
                        self.temperature[x][y] = new_temperature[x][y]

        assert all([
            all([0 < self.temperature[x][y] < 40 for y in range(self.height + 1)]) for x in range(self.width + 1)
        ])
        self.temperature_fix = np.full((self.height + 1, self.width + 1), False)

    def reset(self):
        self.temperature = np.full((self.height + 1, self.width + 1), self.background_temperature, dtype=float)
        self.temperature_fix = np.full((self.height + 1, self.width + 1), False)
        for user in self.users:
            user.reset()
        for service in self.services:
            service.reset()

    def has_history(self, train: bool, iteration: int, steps: List[int]) -> bool:
        history = self.training_history if train else self.testing_history
        return iteration in history and all([step in history[iteration] for step in steps])

    def freeze(self, train: bool, iteration: int, step: int):
        history = self.training_history if train else self.testing_history
        if iteration not in history:
            history[iteration] = {}
        history[iteration][step] = {
            e: e.freeze() for e in self.users + self.services
        }

    def resume(self, train: bool, iteration: int, step: int):
        history = self.training_history[iteration] if train else self.testing_history[iteration]
        for e in self.users + self.services:
            e.resume(history[step][e])

    @property
    def num_users(self) -> int:
        return len(self.users)

    @property
    def num_services(self) -> int:
        return len(self.services)

    @property
    def num_walls(self) -> int:
        return len(self.walls)

    def set_users(self, num_users):
        del self.users
        num = np.random.randint(*num_users) if isinstance(num_users, list) else num_users
        self.users = [
            User(
                coordinate=self.get_random_user_coordinate(),
                orientation=Direction(x=None, y=None, z=0),
                uid=i,
                configuration=self.configuration,
            ) for i in range(num)
        ]

    def set_services(self, num_services):
        del self.services
        self.services = []
        for service_type in num_services:
            num = np.random.randint(*num_services[service_type]) if isinstance(
                num_services[service_type], list
            ) else num_services[service_type]
            coordinates = [self.get_random_coordinate() for _ in range(num)]

            while num > 1 and any([
                np.mean(get_distances(coordinates)) < self.width * self.minimum_service_average_distance_ratio,
                np.min(get_distances(coordinates)) < self.minimum_service_distance,
            ]):
                coordinates = [self.get_random_coordinate() for _ in range(num)]

            for i in range(num):
                self.services.append(service_type(
                    coordinate=coordinates[i],
                    orientation=Direction(x=None, y=None, z=0),
                    sid=len(self.services),
                    configuration=self.configuration,
                ))

    def set_agents(self, agents: List[ServiceAgent]):
        assert len(self.services) == len(agents)
        for i, service in enumerate(self.services):
            service.set(self, agents[i])

    def remove_agents(self):
        for service in self.services:
            service.remove_agent()

    def get_available_services(self, service_type):
        return [service for service in self.services if isinstance(service, service_type) and not service.user]

    def get_context(self):
        return {
            "raw": sum([service.state() for service in self.services], []),
            "communication": self.channel,
        }

    def update_communication(self, message: np.array):
        self.channel = message.tolist()

    def step(self):
        for user in self.users:
            user.step(self)

    def count_concurrency(self) -> int:
        return sum([sum([
            1 if isinstance(
                self.services[i], type(self.services[j])
            ) and self.services[i].user and self.services[j].user
            else 0 for j in range(i + 1, self.num_services)
        ]) for i in range(self.num_services)])

    def block(self, a: Vector, b: Vector) -> List[Tuple[Wall, Vector]]:
        a = a.round(self.block_precision_level)
        b = b.round(self.block_precision_level)

        key = a.tuple() + b.tuple()
        if key not in self.block_cache:
            walls = []
            for wall in self.walls:
                intersection = wall.intersect(a, b)
                if intersection is not None:
                    walls.append((wall, intersection))
            self.block_cache[key] = walls
        return self.block_cache[key]

    def outside(self, coordinate: Vector) -> bool:
        return coordinate.x <= 0 or self.width <= coordinate.x or coordinate.y <= 0 or self.height <= coordinate.y

    def render_field(self) -> Image.Image:
        if self.field is None:
            image = Image.new('RGBA', (self.width * self.scale, self.height * self.scale), (250, 250, 250))
            draw = ImageDraw.Draw(image, "RGBA")

            # Services
            for service in self.services:
                # Location
                x, y = service.coordinate.x * self.scale, service.coordinate.y * self.scale
                image.paste(
                    service.icon.resize((int(self.scale / 2), int(self.scale / 2))),
                    (int(x - self.scale / 4), int(y - self.scale / 2)),
                    service.icon.resize((int(self.scale / 2), int(self.scale / 2)))
                )

                # Direction
                draw.line([
                    (x, y),
                    (x + service.orientation.x * self.scale, y + service.orientation.y * self.scale)
                ], fill=(0, 0, 255), width=1)

            # Walls
            for wall in self.walls:
                c = int(250 * (1 - wall.absorption_rate))
                draw.line([(wall.a.x * self.scale,
                            wall.a.y * self.scale),
                           (wall.b.x * self.scale,
                            wall.b.y * self.scale)],
                          fill=(c, c, c), width=int(self.scale / 5))

            self.field = image

        return self.field.copy()

    def render(self, step: int) -> Image.Image:
        assert record_gif

        image = self.render_field()
        draw = ImageDraw.Draw(image, "RGBA")

        for service in self.services:
            if service.user:
                x, y = service.coordinate.x * self.scale, service.coordinate.y * self.scale

                draw.ellipse(xy=(x - self.scale / 8, y - self.scale / 8, x + self.scale / 8, y + self.scale / 8),
                             fill='green')

        # Users
        effectiveness_sum = 0

        for user in self.users:
            if user.active:
                x, y = user.coordinate.x * self.scale, user.coordinate.y * self.scale
                image.paste(
                    user.icon.resize((int(self.scale), int(2 * self.scale))),
                    (int(x - self.scale/2), int(y - self.scale)),
                    user.icon.resize((int(self.scale), int(2 * self.scale))),
                )

                if user.service:
                    effectiveness = user.feedback(self)
                    effectiveness_sum += effectiveness
                    if effectiveness > 0:
                        color = 'green'
                    else:
                        color = 'red'
                    draw.ellipse(xy=(x - self.scale/4, y - self.scale/4, x + self.scale/4, y + self.scale/4),
                                 fill=color)

                    draw.line([(user.coordinate.x * self.scale,
                                user.coordinate.y * self.scale),
                               (user.service.coordinate.x * self.scale,
                                user.service.coordinate.y * self.scale)],
                              fill=color, width=int(self.scale / 10))

                    for wall, intersection in self.block(user.coordinate, user.service.coordinate):
                        ix, iy = intersection.x * self.scale, intersection.y * self.scale
                        draw.ellipse(
                            xy=(ix - self.scale/4, iy - self.scale/4, ix + self.scale/4, iy + self.scale/4),
                            fill=color
                        )

        # Temperature
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                draw.multiline_text(
                    xy=(x * self.scale, y * self.scale),
                    text=f"{self.temperature[x][y]:.1f}",
                    fill=(
                        int(255 * (self.temperature[x][y] - 10) / 20),
                        0,
                        int(255 * (-self.temperature[x][y] + 30) / 20)
                    ),
                    font=self.small_font,
                )

        # Text
        draw.multiline_text(
            xy=(self.scale, self.scale),
            text=f"Step {step}\nEffectiveness {effectiveness_sum:.2f}",
            fill='blue', font=self.font,
        )

        return image

    def render_dead_space(self) -> Image.Image:
        image = self.render_field()
        draw = ImageDraw.Draw(image, "RGBA")

        for x in range(self.width + 1):
            for y in range(self.height + 1):
                location = Coordinate(x, y, self.depth / 2)
                total_blocked = all([self.block(service.coordinate, location) for service in self.services])
                draw.ellipse(xy=(x * self.scale - self.scale / 16, y * self.scale - self.scale / 16,
                                 x * self.scale + self.scale / 16, y * self.scale + self.scale / 16),
                             fill=(255, 0, 0, 255) if total_blocked else (0, 255, 0, 255))
                mobility = self.mobility_map[x][y]
                draw.line([
                    ((x - mobility.x / 4) * self.scale, (y - mobility.y / 4) * self.scale),
                    ((x + mobility.x / 4) * self.scale, (y + mobility.y / 4) * self.scale)
                ], fill='grey', width=1)

        return image

    def render_mobility(self) -> Image.Image:
        image = self.render_field()
        draw = ImageDraw.Draw(image, "RGBA")

        for x in range(self.width + 1):
            for y in range(self.height + 1):
                mobility = self.mobility_map[x][y]
                draw.line([
                    ((x - mobility.x / 4) * self.scale, (y - mobility.y / 4) * self.scale),
                    ((x + mobility.x / 4) * self.scale, (y + mobility.y / 4) * self.scale)
                ], fill='grey', width=1)

        return image

    def render_value_map(self, service: Service):
        image = self.render_field()
        draw = ImageDraw.Draw(image, "RGBA")

        # Marker
        draw.ellipse(xy=(service.coordinate.x * self.scale - self.scale / 8,
                         service.coordinate.y * self.scale - self.scale / 8,
                         service.coordinate.x * self.scale + self.scale / 8,
                         service.coordinate.y * self.scale + self.scale / 8),
                     fill='green')

        # Memory
        for memory in service.agent.memory.memory:
            draw.ellipse(
                xy=(memory["observation"][0] * self.scale - self.scale / 16,
                    memory["observation"][1] * self.scale - self.scale / 16,
                    memory["observation"][0] * self.scale + self.scale / 16,
                    memory["observation"][1] * self.scale + self.scale / 16),
                fill='grey'
            )

        # Q value
        locations = []
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                locations.append(Coordinate(x, y, self.depth / 2))

        Q = service.prediction(
            samples=[{
                "observation": location.vectorize(),
                "context": {"services": {}, "users": {}}
            } for location in locations]
        )

        for i in range(len(locations)):
            heat = int(max(64., min(255., abs(Q[i]) * 5)))
            draw.text(xy=(locations[i].x * self.scale, locations[i].y * self.scale),
                      text=f"{Q[i]:.1f}",
                      fill=(255, 0, 0, heat) if Q[i] < 0 else (0, 255, 0, heat),
                      font=self.small_font)

        return image

    def save_preview(self, path: str):
        self.render_mobility().save(path)

    def get_random_coordinate(self) -> Coordinate:
        return Coordinate(
            x=float(np.around(np.random.random() * self.width, decimals=3)),
            y=float(np.around(np.random.random() * self.height, decimals=3)),
            z=float(np.around(np.random.random() * self.depth, decimals=3))
        )

    def get_random_user_coordinate(self) -> Coordinate:
        direction = np.random.random() * 4
        if direction <= 1:  # Top
            x = float(np.around(np.random.random() * self.width, decimals=3))
            y = float(np.around(np.random.random(), decimals=3))
            momentum = Mobility(radius=1, theta=np.random.uniform(0, np.pi)+np.pi)
        elif direction <= 2:  # Right
            x = float(np.around(np.random.random() + self.width - 1, decimals=3))
            y = float(np.around(np.random.random() * self.height, decimals=3))
            momentum = Mobility(radius=1, theta=np.random.uniform(0, np.pi)+np.pi/2)
        elif direction <= 3:  # Bottom
            x = float(np.around(np.random.random() * self.width, decimals=3))
            y = float(np.around(np.random.random() + self.height - 1, decimals=3))
            momentum = Mobility(radius=1, theta=np.random.uniform(0, np.pi))
        else:  # Left
            x = float(np.around(np.random.random(), decimals=3))
            y = float(np.around(np.random.random() * self.height, decimals=3))
            momentum = Mobility(radius=1, theta=np.random.uniform(0, np.pi)-np.pi/2)

        coordinate = Coordinate(
            x=x, y=y,
            z=float(np.around(clamp(np.random.normal(
                self.configuration.user_height_mean, self.configuration.user_height_std
            ), *self.configuration.user_height_range), decimals=3))
        )
        coordinate.momentum = momentum

        return coordinate
