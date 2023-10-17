import datetime
import logging
import random
import shutil
import gc
import numpy as np
import tensorflow as tf

from settings import Configuration
from environment import Environment
from experiment import Experiment
from utils import info, summary_path


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Log
    logging.basicConfig(filename=summary_path(now=now, filename=f"log_{now}.txt"), level=logging.INFO)

    # Configuration initialization
    configuration = Configuration()
    configuration.save(summary_path(now=now, filename=f"configuration_{now}.txt"))

    # Plot
    shutil.copy('plot.ipynb', summary_path(now=now, filename=f"plot_{now}.ipynb"))

    # Experiment
    info(f">> Experiment code: {now}")
    experiment = Experiment(now=now, configuration=configuration)

    with tf.device('/GPU:1'):
        for index in range(configuration.num_experiments) if isinstance(
                configuration.num_experiments, int
        ) else configuration.num_experiments:
            info(f">> Experiment [{index}] with new environments")

            seed = random.randint(1, 10000)
            if configuration.control_seed:
                info(seed)

            environments = []
            for e in range(configuration.num_environments):
                environment = Environment(e, configuration)
                environment.save_preview(summary_path(now, index, environment.index, f"Env{environment.index:02}.png"))
                num, dead_space, mean_distance = environment.get_service_distribution()
                info(
                    f"Environment {environment.index}: "
                    f"{environment.num_users} users with "
                    f"services {num}, dead space {dead_space}, mean_distance {mean_distance} "
                    f"and {environment.num_walls} walls"
                )
                environments.append(environment)
            assert len(environments) == configuration.num_environments

            for agent_class in configuration.agents:
                if configuration.control_seed:
                    tf.random.set_seed(seed)
                    np.random.seed(seed)

                for environment in environments:
                    configuration.construct_agents(environment, agent_class)

                # TODO handover
                # TODO exploration
                experiment.run(agent_class=agent_class, experiment_index=index, environments=environments)

                for environment in environments:
                    environment.remove_agents()
                tf.keras.backend.clear_session()
                gc.collect()

            experiment.save_results()
            del environments
            gc.collect()
