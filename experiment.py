import time
import tensorflow as tf
import pandas as pd

from typing import List

from settings import record_gif, record_tensorboard_summary, Configuration
from environment import Environment
from utils import info, summary_path, variable_summary, save_gif


class Experiment:
    def __init__(self, now: str, configuration: Configuration):
        self.now = now
        self.service_types = configuration.num_services.keys()

        # Iteration settings
        self.num_training_steps = configuration.num_training_steps
        self.num_testing_steps = configuration.num_testing_steps
        self.num_testing_iterations = configuration.num_testing_iterations

        self.data = []
        self.stat = []

        # Run settings
        self.agent_class = None
        self.index = None

    @property
    def name(self):
        return self.agent_class.__name__.replace('Agent', '')

    def run(self, agent_class, experiment_index: int, environments: List[Environment]):
        self.agent_class = agent_class
        self.index = experiment_index
        info("")
        info(f"<<<<<<<<<< {self.name} >>>>>>>>>>")

        environments[0].services[0].agent.save_model(summary_path(self.now))
        environments[0].services[0].agent.summary()

        train_writer = tf.summary.create_file_writer(
            summary_path(now=self.now, exp=self.index, name=self.name, filename='train')
        ) if record_tensorboard_summary else None

        start = time.time()

        # Set federation target
        agents = sum([[service.agent for service in environment.services] for environment in environments], [])
        for environment in environments:
            for service in environment.services:
                service.agent.set_federation_target(agents)

        # Training
        self.simulate(
            environments=environments, summary_writer=train_writer,
            iteration=0, steps=list(range(self.num_training_steps)),
            train=True
        )

        # Testing
        for i in range(self.num_testing_iterations):
            self.simulate(
                environments=environments, summary_writer=None,
                iteration=i, steps=list(range(self.num_testing_steps)),
                train=False
            )

        info(f"-- Total {(time.time() - start) / 60:.2f} minutes --")

    def simulate(self, environments: List[Environment], summary_writer, iteration: int, steps: List[int], train: bool):
        start = time.time()

        for environment in environments:
            environment.reset()

        has_history = all([
            environment.has_history(train=train, iteration=iteration, steps=steps)
            for environment in environments
        ])
        info(
            f">> {'Train' if train else 'Test'}ing [{iteration}] "
            f"{'reuse' if has_history else 'record new'} "
            f"history ({steps[0]}-{steps[-1]})"
        )

        frames = {
            environment: []
            for environment in environments
        }

        for step in steps:
            reward_list = []
            loss_list = []

            for environment in environments:
                if has_history:
                    environment.resume(train=train, iteration=iteration, step=step)
                else:
                    environment.step()
                    environment.freeze(train=train, iteration=iteration, step=step)
                environment.propagate_temperature()

                effectiveness_sum = 0

                for user in environment.users:
                    observation = user.state()
                    context = environment.get_context()

                    # Requests
                    if user.request:
                        # Selection
                        candidates = environment.get_available_services(user.request.service)
                        if candidates and step + user.request.duration < steps[-1]:
                            selected = user.selection(candidates, observation, context, train=train)
                            user.acquire(selected)
                        user.remove_request()  # Simply reject if no service available

                    # Episodes
                    if user.episode:
                        # Update context
                        if user.service.control(environment) and user.service.agent.enable_communication:
                            environment.update_communication(user.service.communication({
                                "observation": observation,
                                "context": context,
                            }))

                        # Feedback
                        effectiveness = user.feedback(environment)

                        # Record
                        user.episode.add_observation(observation=observation, context=context, reward=effectiveness)
                        effectiveness_sum += effectiveness

                        # Releasing
                        user.service.duration -= 1
                        if user.service.duration <= 0 or not user.active:
                            finished_episode = user.release()
                            finished_episode.summary()
                            if train and finished_episode.service.agent.ready:
                                loss_list.append(finished_episode.service.agent.learn())
                                finished_episode.service.agent.federation()

                    # User invariant
                    assert not user.request and bool(user.service) == bool(user.episode)

                if record_gif:
                    frames[environment].append(environment.render(step))

                reward = effectiveness_sum
                reward_list.append(reward)
                self.record_data(
                    train=train, index=self.index, env=environment.index, agent=self.name, iteration=iteration,
                    step=step, reward=reward,
                    user=len([user for user in environment.users if user.active]),
                    num=environment.num_services,
                    count=sum([service.count for service in environment.services]),
                    provision=len([service for service in environment.services if service.user]),
                    concurrency=environment.count_concurrency(),
                )

            # Rewards and loss over environments for each step
            variable_summary(summary_writer, 'Main', 'reward', step, reward_list)
            variable_summary(summary_writer, 'Learning', 'loss', step, loss_list)

            # Various statistics
            variable_summary(summary_writer, 'Stat', 'count', step, sum([[
                service.count for service in environment.services
            ] for environment in environments], []))
            variable_summary(summary_writer, 'Stat', 'memory', step, sum([[
                service.agent.memory.length() for service in environment.services
            ] for environment in environments], []))
            variable_summary(summary_writer, 'Stat', 'user', step, [
                len([user for user in environment.users if user.active]) for environment in environments
            ])
            variable_summary(summary_writer, 'Stat', 'provision', step, [
                len([service for service in environment.services if service.user]) for environment in environments
            ])
            variable_summary(summary_writer, 'Stat', 'concurrency', step, [
                environment.count_concurrency() for environment in environments
            ])

        info(f"-- {'Train' if train else 'Test'}ed {(time.time() - start) / 60:.2f} minutes --")

        for environment in environments:
            save_gif(frames[environment], summary_path(
                now=self.now, exp=self.index, env=environment.index, name=self.name,
                filename=f"{'Train' if train else 'Test'}{iteration:02}.gif",
            ))

        return True

    def record_data(self, train, index, env, agent, iteration, step, reward, user, num, count, provision, concurrency):
        self.data.append((train, index, env, agent, iteration, step, reward, user, num, count, provision, concurrency))

    def save_results(self):
        try:
            pd.DataFrame(self.data, columns=[
                'Train', 'Experiment', 'Environment', 'Agent', 'Iteration', 'Step',
                'Rewards', 'Users', 'Services', 'Requests', 'Provisions', 'Concurrency',
            ]).to_csv(summary_path(self.now, filename=f'data_{self.now}.csv'))
        except PermissionError:
            print('File not available')
