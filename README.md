# SDDESolarDynamo

This repository presents the work done for the Laboratoy of Computational Physiscs - B (Physics of Data, Cohort of 2023/2024); the authors are Calandra Buonaura Lorenzo, Rossi Lucrezia and Turci Andrea, supervised by Professor Carlo Albert.

## Theoretical background

This project focuses on modeling solar magnetic activity through the Solar Dynamo mechanism, using a probabilistic inference approach based on the simulated annealing–approximate Bayesian computation (sABC) algorithm. The aim is to infer the parameters of a stochastic solar dynamo model that captures key features of the Sun's magnetic cycle, including its variability and quasi-periodic reversals.

### Solar Dynamo

The Solar Dynamo is a theoretical model that explains the generation and evolution of the Sun’s magnetic field. It arises from the interaction between the Sun’s differential rotation and convective motions in its outer layers, leading to the amplification and cyclical reversal of magnetic fields. This process gives rise to the observed 11-year sunspot cycle and other solar phenomena such as flares and coronal mass ejections.

In this project, the dynamo is represented by a simplified stochastic model that incorporates noise to account for uncertainties and irregularities in solar behavior. This allows us to better capture the chaotic and variable nature of solar cycles.

### sABC algorithm

The sABC (simulated annealing–approximate Bayesian computation) algorithm is a Bayesian inference method particularly suited for complex, stochastic models where the likelihood function is intractable or expensive to evaluate, but easy to sample.

In the context of the solar dynamo, sABC is used to infer model parameters by comparing simulated outputs to real or synthetic data. The algorithm works by:
- Generating candidate parameter sets,
- Simulating the model using those parameters,
- Comparing the results to the observed data using a distance metric,
- Accepting or rejecting candidates based on a probabilistic acceptance criterion that evolves over time (simulated annealing).

This approach allows for efficient exploration of the parameter space, especially in problems with significant noise or high-dimensional dynamics.

The implementation of the sABC algorithm in this project builds upon the [SimulatedAnnealingABC.jl](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl) Julia package, which provides a flexible and efficient framework for approximate Bayesian computation via simulated annealing.

For more in-depth theoretical background and algorithmic details, please refer to the [theory](./theory/) directory.

## Code organization

The source code is located in the [src](./src/) directory and is structured across three main files:

1. [`DirUtils.jl`](./src/DirUtils.jl)  
   Contains utility functions for directory and file management, such as creating output folders or handling file paths for results and plots.

2. [`SDDESolarDynamo.jl`](./src/SDDESolarDynamo.jl)  
   Implements the core logic of the Solar Dynamo model, including the stochastic differential equations, simulation routines, and interaction with the sABC inference pipeline.

3. [`VisualizationTools.jl`](./src/VisualizationTools.jl)  
   Provides plotting and data visualization utilities to analyze simulation results, compare them with reference data, and illustrate the inferred dynamics.

## Examples folder

The [examples](./examples/) folder contains several Jupyter notebooks that illustrate the main stages of the analysis and inference process:

1. [`dataset_analysis.ipynb`](./examples/dataset_analysis.ipynb)  
   Performs a preliminary analysis of the input datasets, showing what kind of data are we dealing with and the relevant quantities later used for the analysis.

2. [`synthetic.ipynb`](./examples/synthetic.ipynb) and [`real.ipynb`](./examples/real.ipynb)  
   Apply the sABC inference algorithm to both synthetic and real datasets, showcasing the performance and behavior of the model under different data conditions. The summary statistics considered for this analysis are some specific Fourier components.

3. [`visualization_synthetic.ipynb`](./examples/visualization_synthetic.ipynb) and [`visualization_real.ipynb`](./examples/visualization_real.ipynb)  
   Visualize the results of the inference process, comparing simulated outputs to the input datasets and highlighting dynamical features. In particular, they show the parameter posteriors for the different simulations.

4. [`postprocessing_visualization.ipynb`](./examples/postprocessing_visualization.ipynb)  
   Provides additional plots and summaries to help interpret the inferred parameters and model trajectories. In particular this is useful to analyze wether the simulations have been successful, because it allows to analyze the results of the single particles used in the algorithm.

In addition, the folder contains three other folders:
1. [data](./examples/data/)
   Contains the data on solar activity used for testing the mdoel.

2. [real_data_sim](./examples/real_data_sim/) and [synthetic_data_sim](./examples/synthetic_data_sim/)
   These directories contains the results of the best simulations, with the parameters used and the solutions obtained. They also have already inside some images that are useful to visualize directly the result of the simulations.

## Conclusions
The final analysis of the project can be found in these [slides](./theory/LCP%20_%20Solar%20dynamo.pdf). 
