# MultiFedRL: Efficient Training of Service Agents for Heterogeneous Internet of Things Environments

The Internet of Things (IoT) has gained more attention for enhancing users' daily lives in public spaces by providing services using shareable devices.
However, uncertain factors and other services in the environment may affect the service severely, resulting in users' low satisfaction.
Based on multi-agent reinforcement learning and cluster-based federated learning, autonomous service agents may learn the complex influence of the factors from user feedback without sophisticated modeling and detection processes.
However, conventional approaches are limited in dealing with multiple clustering dimensions of service agents and dynamic environmental contexts affecting the agents.
In this work, we propose _MultiFedRL: the multi-dimension and multi-agent federated reinforcement learning_ for efficient training of service agents in public IoT environments.
First, we suggest a parallel structure of neural networks for multiple clustering dimensions to share parameters independently, solving the limitation of conventional cluster-based federated learning.
Second, we suggest an environment-centric learnable communication protocol for the agents to summarize and interpret physical contexts consisting of static characteristics and dynamic states.
To evaluate MultiFedRL, we developed a simulation framework for IoT services provided to mobile users in public spaces, imitating the user-service interaction based on crucial physics phenomena.
Experimental results show that MultiFedRL increases user satisfaction by 82.9\% and training efficiency by 24.5\% compared to state-of-the-art cluster-based federated learning.
