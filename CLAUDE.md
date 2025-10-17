# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PhD SLAM research project implementing a minimal version of PHD-SLAM using ORB features and probability of detection based on ORBSLAM3 map management heuristics. The system integrates Random Finite Set (RFS) theory with visual SLAM for robust multi-target tracking and mapping.

## Build System

This project uses **CMake with ROS2 (ament_cmake)** as the build system:

- **CMake Version**: 3.20+ required
- **C++ Standard**: C++20
- **Build Type**: RelWithDebInfo by default

### Common Build Commands

```bash
# From workspace root (one level up from this directory)
colcon build --packages-select phd_orbslam_minimal

# Build with debug symbols
colcon build --packages-select phd_orbslam_minimal --cmake-args -DCMAKE_BUILD_TYPE=Debug

# Clean build
rm -rf ../build && colcon build --packages-select phd_orbslam_minimal
```

### Running the Main Executable

```bash
# Source the workspace
source ../install/setup.bash

# Run with EuRoC dataset configuration
ros2 run phd_orbslam_minimal rbphdslam6dSim_euroc --config-file cfg/rbphdslam6dSim_euroc.yaml
```

## Architecture

### Core Components

- **RFS-SLAM Library** (`rfsslam`): Core SLAM algorithms implemented as a static library
- **RB-PHD Filter**: Rao-Blackwellized Probability Hypothesis Density filter for multi-target SLAM
- **ORB Feature System**: Visual feature extraction and matching using ORB descriptors
- **Measurement Models**: 3D stereo ORB-based measurement models with frustum culling
- **Process Models**: 6D odometry models for camera motion prediction

### Key Dependencies

- **GTSAM 4.3+**: Graph-based optimization backend
- **OpenCV 4.4+**: Computer vision and image processing
- **Eigen3**: Linear algebra computations  
- **Sophus**: Lie group operations for 3D transformations
- **Boost**: Various utilities (filesystem, timer, etc.)
- **ROS2 Humble+**: Robot Operating System framework
- **yaml-cpp**: Configuration file parsing

### Directory Structure

- `include/`: Header files organized by functionality
  - `measurement_models/`: Sensor measurement models
  - `misc/`: Utility classes (memory profiling, YAML serialization)
  - `external/`: Third-party headers (ORB extractor, argparse)
- `src/`: Implementation files
  - `rbphdslam6d_euroc.cpp`: Main executable for EuRoC dataset
  - Core algorithm implementations
- `cfg/`: YAML configuration files
- `test/`: Unit tests and test utilities

### Configuration

The system uses YAML configuration files in `cfg/` directory. The main configuration includes:
- Camera calibration parameters (stereo setup)
- ORB feature extraction settings
- Filter parameters (particle count, thresholds)
- Dataset paths and logging options

### Key Classes

- `RBPHDFilter`: Main filter implementation
- `MeasurementModel_3D_stereo_orb`: ORB-based 3D measurements
- `ProcessModel_Odometry6D`: 6DOF motion model
- `Frame`: Image frame with ORB features
- `OrbslamPose`/`OrbslamMapPoint`: ORBSLAM integration classes

## Testing

The project includes unit tests in the `test/` directory. Run tests after building:

```bash
# From build directory
ctest
```

## Data and Logging

- Logs are written to `data/rbphdslam6d/` directory
- The system supports EuRoC dataset format
- Configuration includes timing and results logging options