# RADAR SEGMENTATION ON POINT CLOUDS DATA
## This is my interpretation of the paper: `O. Schumann, M. Hahn, J. Dickmann and C. Wöhler, "Semantic Segmentation on Radar Point Clouds," 2018 21st International Conference on Information Fusion (FUSION), Cambridge, UK, 2018, pp. 2179-2186, doi: 10.23919/ICIF.2018.8455344.`

### In this version, the model is run for only 2 epochs, The `Macro-averaged F1 Score is 0.76`
### Download the radarscenes dataset at: `https://radar-scenes.com/`

#### Helping content on github: `https://github.com/TUMFTM/RadarGNN/tree/main` and `https://github.com/XZLeo/Radar-Detection-with-Deep-Learning/tree/main`

# Summary of Radar Point Cloud Processing and Classification Project

## Overview
This project involves processing radar point cloud data from the RadarScenes dataset, creating point cloud frames, and using deep learning models (PointNet and PointNet++) to classify objects in the radar scenes. The codebase is structured to manage tasks from data loading and preprocessing to model training and evaluation.

## Modules and Functionality

### RadarPointCloud Class
Defines a class to manage and manipulate radar point cloud data, including methods for data cleaning and visualization.

### Enums for Labels
Defines enums (Label and ClassificationLabel) to handle semantic labels and classification labels for radar point clouds, simplifying the process of mapping and categorizing different classes.

### Helper Functions
Provides utility functions to support data transformations and sensor mounting retrieval:

- **`get_mounting(sensor_id, json_path=None)`**: Retrieves the sensor mounting positions.
- **`batch_transform_3d_vector(trafo_matrix, vec)`**: Applies a transformation matrix to a set of vectors.

### Scene and Sequence Classes
Encapsulates individual scenes and sequences of radar measurements:

- **Scene**: Represents a single scene with radar and odometry data.
- **Sequence**: Represents a sequence of radar measurements, usually constructed from a JSON file.

### SceneCollection Class
Implements a collection of data from multiple consecutive scenes, transforming them into a single point cloud frame. This includes methods to process and visualize the combined data.

### PointCloudProcessor Class
Contains static methods for preprocessing radar point clouds by cropping and removing invalid points:

- **`transform(dataset_config, point_cloud)`**: Preprocesses the point cloud based on dataset configuration settings.

### RadarScenesDataset Class
Defines a PyTorch `Dataset` for managing and processing radar point cloud data, including normalization, resampling, and augmentation:

- **`create_point_clouds()`**: Processes all sequences to generate point clouds.
- **`normalize_point_cloud(point_cloud)`**: Normalizes spatial coordinates and other features.
- **`resample_point_cloud(point_cloud)`**: Ensures a fixed size of reflections per point cloud.
- **`remove_nan(point_cloud)`**: Removes NaN values from features.
- **`augment_point_cloud(point_cloud)`**: Applies data augmentation.

### PointNet and PointNet++ Functionality
Includes classes and methods for implementing PointNet and PointNet++ functionalities:

- **PointNetSetAbstraction**: Performs set abstraction in PointNet++ by sampling points and applying MLPs.
- **PointNetSetAbstractionMsg**: Extends set abstraction to handle multiple scales.

### Trainer Class
Implements a training and evaluation framework for the point cloud classification model using PyTorch:

- **`train()`**: Trains the model using k-fold cross-validation, evaluating on validation and test sets.
- **`evaluate(test_loader)`**: Evaluates the model, computing loss, accuracy, and confusion matrix.
- **`plot_confusion_matrix(cm, title, normalize=False)`**: Plots the confusion matrix.

## Example Usage
```python
# Initialize the trainer and start training
trainer = Trainer(model, dataset, criterion, optimizer, num_epochs=30, batch_size=32, num_folds=5)
trainer.train()
