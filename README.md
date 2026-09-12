# Neural Network Framework in C++

A modular and extensible neural network framework implemented in modern C++ for building and training neural networks. This framework is designed to be lightweight, flexible, and easily integrated into other projects.

This framework is designed with a **test-driven approach**, ensuring reliability and correctness through comprehensive testing of its components.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Test-Driven Development](#test-driven-development)
4. [Getting Started](#getting-started)
5. [Framework Components](#framework-components)
6. [Usage](#usage)
7. [Examples](#examples)
8. [Project Structure](#project-structure)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

This framework provides the building blocks for creating and training neural networks in C++. It includes support for dense layers, activation functions, and utilities for working with datasets like MNIST. The framework is designed to be extensible, allowing users to add custom layers, activation functions, and training logic.

## Features

- Layer Support: Includes fully connected (dense) layers and Convolutional layers.
- Activation Functions: Built-in support for ReLU, Sigmoid, and Softmax.
- Dataset Utilities: Functions for loading and preprocessing the MNIST dataset.
- Serialization: Save and load models for reuse.
- Extensibility: Add custom layers, activation functions, or training algorithms.

## Getting Started

### Requirements
- C++17 compiler (g++/clang++/MSVC)
- CMake 3.15+
- Standard library only (no external dependencies)
- Optional: [Ninja](https://ninja-build.org/) (used by the default CMake preset; drop `-G Ninja` / use plain `cmake -B build` if you don't have it)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/ahmedesmailmehanna/Neural-Network-Cpp.git
cd Neural-Network-Cpp
```
2. Include the framework in your project:
  - Add the src directory to your project's include path.
  - Include the necessary headers in your code.

### Compilation (CMake)

This repo is a library (everything under `src/`) plus two thin executables that link against it: `main.cpp` (a usage example) and `tests/test.cpp` (the test suite). Building is done through CMake rather than hand-written compiler invocations:

```bash
# Configure + build (Release, -O3, via CMakePresets.json)
cmake --preset default
cmake --build --preset default

# Run the example (accuracy test for model v3.1)
./build/main

# Run the tests
./build/test
# or via CTest:
ctest --preset default
```

If Ninja isn't installed, configure without a preset instead:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

To build only the library (e.g. when consuming this repo from another CMake project), disable the example and tests:
```bash
cmake --preset default -DNNCPP_BUILD_EXAMPLE=OFF -DNNCPP_BUILD_TESTS=OFF
```

See `CMakeLists.txt` for the full set of options, and `notes.txt` for the equivalent commands plus the legacy raw `g++` invocation kept for reference.

## Framework Components

### Core
- NeuralNetwork: A template class for managing layers and training the network.
- Trainable: An interface for trainable components.
- Serializable: An interface for saving and loading models.

### Layers
- DenseLayer: A fully connected layer with customizable activation functions.
- ConvLayer: ConvLayer: A convolutional layer supporting filters, strides, padding, and activation functions.
- More to be added...

### Activations
- ReLUFunction: Rectified Linear Unit activation
- SigmoidFunction: Sigmoid activation.
- SoftmaxFunction: Softmax activation for output layers.

### Utilities
- utils: Functions for loading MNIST images and labels, flattening matrices, and creating target matrices.

## Usage

### Creating a Neural Network
1. Define the network structure:
```c++
NeuralNetwork nn;
nn.addLayer(std::make_unique<DenseLayer>(784, 16, new activations::Sigmoid()));
nn.addLayer(std::make_unique<DenseLayer>(16, 16, new activations::Sigmoid()));
nn.addLayer(std::make_unique<DenseLayer>(16, 10, new activations::Softmax(), true)); // Output layer must be identified with a true flag
```
2. Load and preprocess the dataset:
```c++
std::vector<Matrix> input = utils::loadMNISTImages("./data/train-images-idx3-ubyte");
for (auto& img : input) {
    img = utils::flatten(img);
}
std::vector<int> labels = utils::loadMNISTLabels("./data/train-labels-idx1-ubyte");
std::vector<Matrix> target(labels.size());
for (int i = 0; i < labels.size(); i++) {
    target[i] = utils::createMNISTTargetMatrix(labels[i]);
}
```
3. Train the network:
```c++
nn.train(input[0], target[0], 10, 0.01);
// nn.train(Matrix &input, Matrix &target, int epochs, double learning_rate);
```
4. Save the trained model:
```c++
nn.saveToFile("./models/model_v1");
```

## Examples

Example 1: Training a Neural Network
```c++
NeuralNetwork nn;
nn.addLayer(std::make_unique<DenseLayer>(784, 16, new activations::Sigmoid()));
nn.addLayer(std::make_unique<DenseLayer>(16, 16, new activations::Sigmoid()));
nn.addLayer(std::make_unique<DenseLayer>(16, 10, new activations::Softmax()));

std::vector<Matrix> input = utils::loadMNISTImages("./data/train-images-idx3-ubyte");
for (auto& img : input) {
    img = utils::flatten(img);
}

std::vector<int> labels = utils::loadMNISTLabels("./data/train-labels-idx1-ubyte");
std::vector<Matrix> target(labels.size());
for (int i = 0; i < labels.size(); i++) {
    target[i] = utils::createMNISTTargetMatrix(labels[i]);
}

for (int i = 0; i < target.size(); i++) {
  std::cout << "Trianing number: " << i << std::endl;
  nn.train(input[i], target[i], 200, 0.01);
}
nn.saveToFile("./models/model_v1");
```

Example 2: Loading and Testing a Model
```c++
nn.loadFromFile("./models/model_v1");
Matrix output = nn.forward(input[123]);
output.print();
input[123].print();
```

## Project Structure

```
Neural-Network-Cpp/
├── src/
│   ├── activations/         # Activation functions (ReLU, Sigmoid, Softmax)
│   ├── core/                # Core components (NeuralNetwork, Trainable, Serializable)
│   ├── layers/              # Layer implementations (DenseLayer, ConvLayer)
│   ├── math/                # Matrix operations and utilities
│   ├── utils/               # MNIST loader and utility functions
├── tests/                   # Unit tests for the framework
├── data/                    # MNIST dataset files
├── CMakeLists.txt           # Build definition (library + example + tests)
├── CMakePresets.json        # Ninja/Release and Debug presets
├── main.cpp                 # Usage example (not part of the library itself)
├── notes.txt                # Build commands + development notes
├── README.md                # Project documentation
```

## Contributing

Contributions are welcome! To contribute:
  1. Fork the repository.
  2. Create a new branch for your feature or bug fix.
  3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
