# Chess Pieces Detection

## Introduction
The **Chess Pieces Detection** project is a comprehensive computer vision system designed to automatically detect chessboards and chess pieces from images or video streams. It maps the detected pieces to their corresponding squares on the board and extracts the game state into a standard FEN (Forsyth-Edwards Notation) string, which can then be visualized in a 2D interface. This project streamlines the process of digitizing physical chess games for analysis, broadcasting, or record-keeping.

## Project Organization
The repository is organized as follows:

- `configs/`: Contains configuration files for model parameters, inference settings, and pipelines.
- `datasets/`: Stores the datasets used for training and evaluating the object detection models.
- `src/`: Core source code and modules for the detection, mapping, and extraction logic.
- `scripts/`: Executable scripts for running training, evaluation, and end-to-end inference workflows.
- `runs/`: Output directories for model training checkpoints, logs, and inference results.
- `cloud_computing/`: Scripts and configurations for cloud-based execution and deployment.
- `pyproject.toml` : Python project and dependency configurations.

## Acknowledgments & References
This project builds upon and integrates excellent work from the open-source community. We would like to express our gratitude to the following references:

- **AI_Chess**: We referenced and adapted their robust models for accurate chess pieces and chessboard detection.
- **Dynamic-Chess-Board-Piece-Extraction**: We utilized their comprehensive pipeline logic, which covers detecting board cells and pieces, mapping the pieces to their correct cells, and exporting the final FEN string along with the 2D visualizer.
