# MultiFedRL: Efficient and Specialized Training for Service Agents in Heterogeneous Web of Things Environments

The Web of Things (WoT) has gained more attention for enhancing users' daily lives, providing various services using distributed devices that sense and actuate physical environments.
To adapt services to diverse environments, autonomous agents that provide services may actively learn from user interactions.
In this process, the agents suffer from sparse user feedback and variant environments, resulting in limited and heterogeneous data.
Using cluster-based federated learning and further fine-tuning, the agents can collaboratively share learned knowledge and specialize to each environment without revealing private data.
However, conventional cluster-based federated learning cannot deal with two distinct dimensions of clusters in WoT: environment and service. 
Moreover, the agents require effective communication to share dynamic environmental contexts.
In this work, we propose MultiFedRL: the multi-dimension and multi-agent federated reinforcement learning for efficient and asynchronous training of service agents in WoT environments.
First, MultiFedRL deals with multiple dimensions of clusters through a parallel and modular architecture of neural networks to independently share each partition.
Second, MultiFedRL uses a learnable communication protocol of the agents to summarize and interpret physical contexts consisting of environment-specific characteristics and actuation states of services.
Experimental results in simulations show that MultiFedRL outperforms other state-of-the-art cluster-based federated learning in terms of collected rewards and training speed.
