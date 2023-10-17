import tensorflow as tf
import numpy as np
import os

from typing import List

from utils import ExperienceMemory, info


class ServiceAgent:
    def __init__(
            self, name,
            layer_specs, user_state_size: int, service_state_size: int,
            memory_size: int, optimizer, learning_rate: float, minimum_batch: int,
            averaging: bool, federation_rate: float,
            context_raw_size: int, embedding_size: int,
    ):
        self.name = name if name is not None else self.__class__.__name__
        self.env = None
        self.type = None

        # Neural network
        self.user_state_size = user_state_size
        self.service_state_size = service_state_size
        self.embedding_size = embedding_size
        self.context_size = embedding_size if self.enable_communication else context_raw_size
        self.layer_specs = layer_specs

        # Learning
        self.memory = ExperienceMemory(memory_size)
        self.minimum_batch = minimum_batch
        # Optimizer
        self.prediction_optimizer = optimizer(learning_rate=learning_rate)
        self.communication_optimizer = optimizer(learning_rate=learning_rate)

        # Federation
        self.federation_target = {}
        self.federation_rate = federation_rate
        self.averaging = averaging

        # Model
        self.model, self.communication_model = self.build_model()

    def set(self, env, service):
        self.env = env.index
        self.type = service.__class__.__name__

    @property
    def enable_communication(self):
        return False

    def build_model(self) -> (tf.keras.Model, tf.keras.Model):
        observation = tf.keras.Input(shape=(self.user_state_size,), dtype=tf.float32, name='observation')
        service_state = tf.keras.Input(shape=(self.service_state_size,), dtype=tf.float32, name='service_state')
        context = tf.keras.Input(shape=(self.context_size,), dtype=tf.float32, name='context')

        x = tf.concat([observation, service_state, context], axis=-1)
        x = tf.keras.layers.Dense(self.embedding_size, activation=None)(x)
        x = self.build_hidden_layers(x)

        return tf.keras.Model(
            name='Q',
            inputs=[observation, service_state, context],
            outputs=[tf.keras.layers.Dense(units=1, activation='tanh', name='Q')(x)],
        ), tf.keras.Model(
            name='communication',
            inputs=[observation, service_state, context],
            outputs=[tf.keras.layers.Dense(self.context_size, activation='tanh', name='environment_communication')(x)]
        ) if self.enable_communication else None

    def build_hidden_layers(self, x):
        for group in self.layer_specs:
            for i, spec in enumerate(self.layer_specs[group]):
                x = spec.instantiate(name=f"{group}_{i}")(x)
        return x

    def summary(self):
        if self.model:
            self.model.summary(print_fn=info)
        if self.communication_model:
            self.communication_model.summary(print_fn=info)

    def prediction(self, samples, training):
        return self.model({
            "observation": samples["observation"],
            "service_state": np.zeros_like(samples["service_state"]),
            "context": samples["context"],
        }, training=training)

    def communication(self, samples, training):
        return self.communication_model({
            "observation": np.zeros_like(samples["observation"]),
            "service_state": samples["service_state"],
            "context": samples["context"],
        }, training=training)

    def set_federation_target(self, agents):
        pass

    def federation(self):
        for group in self.federation_target:
            federation_target = np.random.choice(
                self.federation_target[group],
                size=int(len(self.federation_target[group]) * self.federation_rate),
                replace=False,
            )
            weights = [[
                np.mean(weight, axis=0) for weight in zip(*layer)
            ] for layer in zip(*[
                agent.get_sharable_weights(group) for agent in federation_target
            ])] if self.averaging else self.get_sharable_weights(group)

            for agent in federation_target:
                agent.set_sharable_weights(group, weights)

    def get_sharable_weights(self, group):
        return [layer.weights for layer in self.model.layers if group in layer.name]

    def set_sharable_weights(self, group, weights):
        for i, layer in enumerate([layer for layer in self.model.layers if group in layer.name]):
            layer.set_weights(weights[i])

    def add(self, new_experiences: List[dict]):
        for experience in new_experiences:
            self.memory.add(
                observation=experience["observation"],
                service_state=experience["service_state"],
                context=experience["context"]["communication" if self.enable_communication else "raw"],
                reward=experience["reward"],
            )

    @property
    def ready(self):
        return self.memory.length() >= self.minimum_batch

    def learn(self):
        samples = self.memory.sample()

        # Q prediction loss
        with tf.GradientTape() as tape:
            prediction = self.prediction(samples, training=True)
            prediction_loss = tf.reduce_mean(tf.square(prediction - samples["reward"]))

        self.prediction_optimizer.apply_gradients(zip(
            tape.gradient(prediction_loss, self.model.trainable_weights),
            self.model.trainable_weights
        ))

        # Communication loss
        if self.enable_communication:
            with tf.GradientTape() as tape:
                prediction = self.prediction({
                    "observation": samples["observation"],
                    "service_state": samples["service_state"],
                    "context": self.communication(samples, training=True),
                }, training=True)
                communication_loss = -tf.reduce_mean(prediction)

            self.communication_optimizer.apply_gradients(zip(
                tape.gradient(communication_loss, self.communication_model.trainable_weights),
                self.communication_model.trainable_weights
            ))

            return prediction_loss + communication_loss

        return prediction_loss

    def save_model(self, path):
        try:
            if self.model:
                tf.keras.utils.plot_model(
                    self.model,
                    to_file=os.path.join(path, f"{self.__class__.__name__}.png"),
                    show_shapes=True, expand_nested=True, show_layer_activations=True, show_layer_names=True,
                    dpi=300, rankdir='TB'
                )
            if self.communication_model and self.enable_communication:
                tf.keras.utils.plot_model(
                    self.communication_model,
                    to_file=os.path.join(path, f"{self.__class__.__name__}_communication.png"),
                    show_shapes=True, expand_nested=True, show_layer_activations=True, show_layer_names=True,
                    dpi=300, rankdir='TB'
                )
        except ImportError as e:
            info(e)


class RandomAgent(ServiceAgent):
    def prediction(self, samples, training):
        return tf.random.uniform(shape=(1, 1))

    @property
    def ready(self):
        return False

    def build_model(self):
        return None, None


class IndependentAgent(ServiceAgent):
    pass


class ClusterFedAgent(ServiceAgent):
    def set_federation_target(self, agents):
        self.federation_target = {
            "environment": [agent for agent in agents if agent.env == self.env and agent.type == self.type],
            "service": [agent for agent in agents if agent.env == self.env and agent.type == self.type],
        }


class FlexiFedAgent(ServiceAgent):
    def set_federation_target(self, agents):
        self.federation_target = {
            "environment": [agent for agent in agents if agent.env == self.env],
            "service": [agent for agent in agents if agent.env == self.env and agent.type == self.type],
        }


class MultiFedAgent(ServiceAgent):
    def build_hidden_layers(self, x):
        environment_x = x
        for i, spec in enumerate(self.layer_specs["environment"]):
            environment_x = spec.instantiate(name=f"environment_{i}")(environment_x)

        service_x = x
        for i, spec in enumerate(self.layer_specs["service"]):
            service_x = spec.instantiate(name=f"service_{i}")(service_x)

        x = tf.concat([environment_x, service_x], axis=-1)
        for i, spec in enumerate(self.layer_specs["agent"]):
            if 'units' in spec.kwargs:
                x = spec.instantiate(name=f"agent_{i}", units=spec.kwargs['units']/2)(x)
            else:
                x = spec.instantiate(name=f"agent_{i}")(x)

        return x

    def set_federation_target(self, agents):
        self.federation_target = {
            "environment": [agent for agent in agents if agent.env == self.env],
            "service": [agent for agent in agents if agent.type == self.type],
        }


class MultiFedRLAgent(MultiFedAgent):
    def enable_communication(self):
        return True
