import tensorflow as tf
import json

from agent import ServiceAgent, RandomAgent
from agent import IndependentAgent, ClusterFedAgent, FlexiFedAgent
from agent import MultiFedRLAgent
from models import SpeakerService, DisplayService, CoolerService


"""
units
distance = m
sound_pressure = dB
time = minute
"""

record_gif = False
record_tensorboard_summary = True


class Configuration:
    def __init__(self):
        # Experiment
        self.control_seed = True
        self.num_experiments = 10
        self.num_environments = 5
        self.num_training_steps = 1 * 1000  # minutes
        self.num_testing_steps = 1 * 1000
        self.num_testing_iterations = 1

        # Environment
        self.width = 15
        self.height = 15
        self.depth = 3

        self.wall_length_ratio = [0.1, 0.5]
        self.minimum_service_average_distance_ratio = 0.6
        self.minimum_service_distance = 2
        self.temperature_propagation_iterations = 10
        self.block_precision_level = None

        self.num_users = [5, 15]
        self.num_services = {
            SpeakerService: [10, 20],
            DisplayService: [10, 20],
            CoolerService: [5, 15],
        }
        self.num_walls = 5

        self.background_noise = 30
        self.background_temperature = 20

        # Service model
        self.speaker_adjust_step = 1
        self.speaker_maximum_intensity = [60, 100]
        self.display_adjust_step = 100
        self.display_maximum_text_size = [500, 1500]
        self.cooler_maximum_range = 5
        self.cooler_maximum_distance = [1, 10]
        self.cooler_maximum_adjust_step = [1, 10]

        # User model
        self.user_height_range = [1.5, 1.9]
        self.user_height_mean = 1.7
        self.user_height_std = 0.1
        self.user_duration_range = [1, 5]
        self.user_enter_probability = 0.1
        self.user_exit_probability = 0.01
        self.user_request_probability = 0.1
        self.user_feedback_probability = 1
        self.user_feedback_dense = False
        self.user_effectiveness_scale = 1
        self.user_speed_scale = 1
        self.momentum_ratio = 0.2

        self.user_visual_acuity = 0.0
        self.user_visual_field_max = 80
        self.user_viewing_angle_max = 70
        self.user_acoustic_acuity = 40  # Minimum sound pressure to hear
        self.user_temperature_expected = [10, 20]
        self.user_temperature_buffer = 2  # Acceptable difference of degree

        self.exploration = True

        # Wall model
        self.wall_sound_absorption_rate_range = [0.9, 1.0]

        # Agent
        self.agents = [
            RandomAgent,
            IndependentAgent,
            ClusterFedAgent,
            FlexiFedAgent,
            MultiFedRLAgent,
        ]
        self.embedding_size = 32
        self.agent_spec = Spec(
            layer_specs={
                "environment": [
                    Spec(tf.keras.layers.Dense, units=self.embedding_size, activation='relu', kernel_regularizer='l2'),
                ],
                "service": [
                    Spec(tf.keras.layers.Dense, units=self.embedding_size, activation='relu', kernel_regularizer='l2'),
                ],
                "agent": [
                    Spec(tf.keras.layers.Dense, units=self.embedding_size, activation='relu', kernel_regularizer='l2'),
                ]
            },
            user_state_size=3,
            memory_size=None, optimizer=tf.keras.optimizers.Adam, learning_rate=1e-2,
            minimum_batch=1,
            # Federation
            averaging=True, federation_rate=0.2,
            # Communication
            embedding_size=self.embedding_size,
        )
        self.initialize_agents_identically = True

    def construct_agents(self, env, agent_class: ServiceAgent):
        agents = [
            self.agent_spec.instantiate(
                agent_class,
                service_state_size=len(env.services[i].state()),
                context_raw_size=len(env.get_context()["raw"]),
            ) for i in range(env.num_services)
        ]
        if self.initialize_agents_identically and agents[0].model:
            for agent in agents:
                agent.model.set_weights(agents[0].model.get_weights())
        env.set_agents(agents)
        return agents

    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            f.write(json.dumps(self, indent=4, cls=CustomJSONEncoder))
            f.close()


class Spec:
    def __init__(self, constructor=None, **kwargs):
        self.constructor = constructor
        self.kwargs = kwargs

    def instantiate(self, constructor=None, name=None, **kwargs):
        self.constructor = constructor if constructor is not None else self.constructor
        k = self.kwargs.copy()
        k.update(kwargs)
        return self.constructor(name=name, **k)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Configuration):
            return {
                key: obj.__dict__[key]
                if not isinstance(obj.__dict__[key], dict)
                else {k.__name__: obj.__dict__[key][k] for k in obj.__dict__[key]}
                for key in obj.__dict__
            }
        if isinstance(obj, Spec):
            return obj.__dict__
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)
