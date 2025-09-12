# LatticeAttentionNet: Variational Monte Carlo with Attention Neural Quantum States

This project is a Julia-based framework for performing Variational Monte Carlo (VMC) simulations to study quantum many-body systems. It utilizes Neural Quantum States (NQS) to represent the wavefunctions of these systems.

## Project Structure

```
/
├── src/                  # Source code for the VMC simulation
├── test/                 # Tests for the different components
├── pretrain/             # Scripts for pre-training models
├── examples/             # Usage examples (currently empty)
├── *.bson                # Saved model weights
└── Readme.md             # This file
```

## Core Components

The `src` directory contains the core components of the VMC simulation:

*   **`VMC.jl`**: The main module that orchestrates the VMC simulation.
*   **`wavefunctions.jl`**: Defines the neural network architectures used as wavefunctions, such as SlaterNet.
*   **`hamiltonians.jl`**: Implements the Hamiltonians for various physical systems.
*   **`sampling.jl`**: Contains the Monte Carlo sampling algorithms, such as the Metropolis-Hastings algorithm, for sampling configurations from the wavefunction's probability distribution.
*   **`optimizer.jl`**: Implements optimization algorithms, like ADAM, to train the parameters of the neural network wavefunction.
*   **`observables.jl`**: Provides functions to calculate physical observables, such as energy and other properties of the system.
*   **`hartree_fock.jl`**: Implements the Hartree-Fock method.
*   **`utils.jl`**: A collection of utility functions used throughout the project.

## Pre-training

The `pretrain` directory contains scripts for pre-training the neural network wavefunctions. This can help to initialize the models in a good region of the parameter space, leading to more stable and efficient VMC calculations.

## Usage

To run a VMC simulation, you would typically define a physical system by specifying a Hamiltonian and a wavefunction, and then use the `VMC.jl` module to run the simulation.

*(Note: The `examples` directory is currently empty. Please refer to the source code and tests for usage patterns.)*

## Tests

The project includes tests in the `test` directory to ensure the correctness of the implemented components. The current tests cover:

*   **`test_slater.jl`**: Tests for the Slater determinant component of the wavefunction.
*   **`test_slaternet.jl`**: Tests for the SlaterNet wavefunction.
*   **`test_attention.jl`**: Tests for the attention mechanism used in some wavefunctions.
*   **`test_delogdet.jl`**: Tests for the custom derivative of the log-determinant.
*   **`test_wf.jl`**: General tests for the wavefunction.

To run the tests, you can execute the test files using the Julia test framework.
