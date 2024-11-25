# Dog Breed Classification

A deep learning-based image classification project that identifies 120 different dog breeds using TensorFlow and TensorFlow Hub. The project processes over 10,000 labeled images to train a robust multi-class classification model.

---

## Features

- **Image Preprocessing**: Converts images into tensors and standardizes sizes.
- **Custom Data Batching**: Efficiently handles large datasets using TensorFlow's data pipeline.
- **Model Training and Validation**: Implements transfer learning with TensorFlow Hub's pre-trained MobileNetV2 model.
- **Visualization Tools**: Provides data insights and predictions through visualization functions.
- **Performance Tracking**: Supports TensorBoard for monitoring training progress.
- **Model Persistence**: Enables saving and loading models for reusability.

---

## Prerequisites

Before running the program, ensure you have the following:

1. **Python Environment**:
   - Python 3.x installed.

2. **Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install tensorflow tensorflow-hub pandas matplotlib scikit-learn
   ```

3. **Dataset**:
   - Images and labels in the required structure.
   - The CSV file `labels.csv` containing image IDs and their corresponding breeds.

## How to Use

### 1. Data Preparation

- Store images in a directory (`/Data/train/`) accessible to the script.
- Use the `labels.csv` file to map image IDs to corresponding breeds.

### 2. Running the Script

Run the script in an environment with access to the dataset:
```bash
python dogbreed.py
```

### 3. Key Operations

- **Data Exploration**: Provides a detailed summary of the dataset.
- **Training the Model**: Uses pre-trained MobileNetV2 to classify dog breeds.
- **Model Evaluation**: Evaluates performance on validation data.

---

## File Structure

- **dogbreed.py**: Main script for processing, training, and evaluating the model.
- **/Data/train/**: Directory containing training images.
- **/Data/labels.csv**: CSV file mapping image IDs to breed labels.

---

## Key Functions

1. **`process_image(image_path)`**:
   - Converts an image to a TensorFlow tensor of size (224x224).
   
2. **`create_data_batches(x, y, ...)`**:
   - Creates batches for training, validation, and testing.

3. **`create_model(input_shape, output_shape, model_url)`**:
   - Builds a model using a pre-trained TensorFlow Hub module.

4. **`train_model()`**:
   - Trains the model and applies early stopping to optimize performance.

5. **Visualization Functions**:
   - `show_25_images(images, labels)`: Displays a batch of images with labels.
   - `plot_pred(...)`: Visualizes predictions and confidence scores.

---

## Results

- The model achieves high accuracy in identifying breeds across diverse images.
- Predictions are made with confidence scores for each breed.

---

## Authors

- Avni Tongia

---
