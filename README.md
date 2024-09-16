# Connect-4 Game AI Agents Training

  

This repository contains code and models for training AI agents to play the Connect-4 game using Deep Q-Learning. The agents are trained against each other or random moves to improve their performance over multiple epochs.

  

## Project Structure

  

-  **`connect_four_env.py`**: Environment setup for the Connect-4 game.

-  **`dqn_agent.py`**: Defines the Deep Q-Network (DQN) agent class used for training.

-  **`train_agents.py`**: Script to train individual agents against random moves.

-  **`train_agents_eachother.py`**: Script to train two agents against each other.

-  **`test_agent.py`**: Script to test a trained agent.

-  **`test_agents_eachother.py`**: Script to test two trained agents against each other.

-  **`play_with_agent.py`**: Allows a human to play against a trained agent.

-  **`requirements.txt`**: Lists the Python dependencies required to run the project.

  

## Model Files

  

Model files (`*.h5`) are generated and saved after every 100 epochs during training. These files are not tracked in this repository and are ignored using the `.gitignore` file.

  

## Usage

  

### Training the Agents

  

To train the agents, use the following commands:

  

1.  **Train an agent against random moves:**

```bash

python train_agents.py

```

  

2.  **Train two agents against each other:**

```bash

python train_agents_eachother.py --agent1 agent1 --agent2 agent2 --episodes 1000

```

  

### Testing the Agents

  

To test the agents' performance, use the following command:

  

```bash

python  test_agents_eachother.py  --agent1  agent1  --agent2  agent2  --episodes  100

```

  

### Installing Requirements

  

To install the required dependencies, run:

  

```bash

pip  install  -r  requirements.txt

```

  

## Contributing

  

Feel free to open issues or submit pull requests if you have any suggestions or improvements.

  

