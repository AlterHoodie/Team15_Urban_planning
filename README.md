# SmartScape : AI Powered Sustainable Urban Planning
![High Level Design	](/assets/HighLevelDesign.jpeg "Model Overview")
---

This project addresses the need for sustainable urban planning in rapidly growing cities through an AI-driven framework that optimizes land use, infrastructure, and traffic management. Guided by principles like the "15-minute city", the framework integrates diverse factors such as geology, current land usage, and population demographics to create accessible, human-centric urban layouts. It prioritizes sustainability by addressing essential urban elements, including road networks, drainage systems, and waste management, while promoting inclusivity and livability. Objectives:
Equitable Urban Space Allocation: Ensure fair and inclusive zoning for residential, commercial, and public spaces by incorporating socio-economic and demographic insights to guide development that meets diverse community needs.
Efficient Connectivity and Traffic Flow: Design efficient road and transit to improve connectivity, reduce congestion, and promote environmentally friendly modes of travel, such as walking and cycling.
Sustainable Land Use and Resource Management: Maximize the utility of land and resources by incorporating geological data, current usage patterns, and hydrological systems. Develop robust drainage frameworks and implement eco-conscious waste management solutions to create a resilient urban ecosystem.

## Installation 

### Environment
* **Tested OS:** Linux
* Python >= 3.8
* PyTorch >= 1.8.1, <= 1.13.0
### Dependencies:
1. Install the required libraries
	```
	pip install -r requirements.txt
	```
2. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
3. Set the following environment variable to avoid problems with multiprocess trajectory sampling:
    ```
    export OMP_NUM_THREADS=1
    ```
   
## Data Processing	
JOSM is a powerful, open-source editor for OpenStreetMap (OSM) data. It is designed for users who need to edit and manage map data in detail.
For this project, we have chosen a specific region in Sarjapur as our dataset.

### Additional Data Sources:
To enhance our analysis, we integrate the following publicly available datasets:
BWSSB Water Lines 
BWSSB Sewer Lines

### Integration:
We overlay these datasets on the OSM base map to create a comprehensive, consolidated map.

### Conversion to GeoPandas DataFrame:
We convert the .osm file, which contains the mapped data, into a GeoPandas DataFrame. This transformation enables us to leverage geospatial operations and analyses within the GeoPandas framework.
### Sample GDF for model
![Sample GDF](/assets/GDF.jpeg "Sammple GDF")

### Tag Standardization:
We standardize the existing OSM tags into a uniform format. This ensures that the data is compatible with our model and can be accurately interpreted during subsequent analysis.

The figure below illustrates the plot annotated on the JOSM editor.
![Loading Data Overview](/assets/Unprocessed.jpeg "Initial Conditions")

The figure below illustrates the preprocessed plot layout.
![Loading Data Overview](/assets/Preprocessed.jpeg "Processed Data")


## Training
You can train your own models using the provided config in [urban_planning/cfg/exp_cfg/real](urban_planning/cfg/exp_cfg/real).

For example, to train a model for the community 1, run:
```
python3 -m urban_planning.train --cfg hlg --global_seed 111
```
You can replace `hlg` to `dhm` to train for the community 2.

## Visualization
You can visualize the generated spatial plans using the provided notebook in [read_output](/read_output/original_plot.ipynb).

## Baselines
To evaluate the centralized heuristic, run:
```
python3 -m urban_planning.eval --cfg hlg --global_seed 111 --agent rule-centralized
```

To evaluate the decentralized heuristic, run:
```
python3 -m urban_planning.eval --cfg hlg --global_seed 111 --agent rule-decentralized
```

You can replace `hlg` to `dhm` to evaluate for the community 2.

## Brief explaination of important files
### The urban environment is modeled through several interconnected components that simulate real-world urban dynamics:
#### city.py: 
The module which carries the physical and socio-economic attributes of the city. It defines elements such as:
1. Geographical Layout: The geographical distribution of districts, neighborhoods, main facilities (roads, parks, public transport).
2. Dynamic Variables: Population density alteration, change of land use type, the alteration of environmental parameters that vary over time as the result of the decision-making process of the agent and other factors such as the economic or demographic shifts.
3. Inter-agent Interactions: The approaches for modeling how many agents or forces inside the city might communicate and behave, and offering a genuine image of urban systems.
   
#### city_config.py:
This configuration module includes the possibility of setting the following simulation parameters:
1. Initial Conditions: Define the initial conditions of the city which include initial population size in the city and initial resource base and initial infrastructure.
2. Simulation Parameters: Explain the number of periods in a simulation process and the frequency of decision making intervals and other operational parameters in order to manage the simulation and its length.

#### plan_client.py:
This module assists in communication of the urban planning agent with the surrounding city environment. It acts as an intermediary, managing the flow of information through:
1. Observation Relay: Informing the state of the city to the agent for assessment.
2. Action Execution: Appling the actions defined by the agent to the city model and changing the environment according to the results of performed steps.

#### observation_extractor.py: 
This component is very important in the process of data collection and data cleaning. It extracts relevant features from the city environment, such as:
1. Traffic Data: Information on the state of traffic congestion, traffic intensity, and the use of public transportation in real-time.
2. Resource Distribution: Information concerning the accessibility and consumption of the basic necessities (water, energy) in various zones of the city.

#### agent.py:
The provided code defines an UrbanPlanningAgent class, an extension of a reinforcement learning agent, tailored for urban planning tasks within a custom environment, CityEnv. It integrates both policy and value networks for decision-making.
1. The agent supports various architectures, including graph-based models (SGNN), MLPs, and rule-based baselines, to simulate and optimize urban plans, leveraging reinforcement learning techniques like PPO.
2. Advanced functionalities include policy optimization, reward evaluation, model checkpointing, and environment-specific adaptations like freezing road or land-use configurations.
3. Logging, visualization, and trajectory sampling tools are incorporated to track performance and generate interpretable plans, with optional video generation to demonstrate iterative planning processes.

