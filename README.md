# audio-tagging-of-entity

A machine learning project for classifying and tagging audio files (such as cat, dog, and human sounds) using deep learning models. This project implements both Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) for audio classification tasks.

## Project Structure

- `Audios/` — Contains audio files organized by category (cat, dog, human).
- `models/` — Stores trained model files and model checkpoints.
- `model.ipynb` / `testL.ipynb` — Jupyter notebooks for training and testing models.
- `audio_labels_with.csv`, `audio_labels.csv`, `audio_labels_length.csv` — CSV files containing audio metadata, labels, and duration information.
- `cfg.py` — Configuration file for model parameters and settings.
- `count_entity.py` — Utility script for entity counting and analysis.

## Features

- Audio classification for multiple entities (cat, dog, human).
- Deep learning models: 
  - Convolutional Neural Networks (CNN) for spatial feature extraction
  - Recurrent Neural Networks (RNN/LSTM) for temporal feature analysis
- Data augmentation and preprocessing utilities:
  - MFCC feature extraction
  - Audio normalization
  - Random sampling and padding
- Jupyter notebooks for experimentation and visualization
- Model checkpointing and best model saving
- Class distribution analysis and visualization

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aelishkumar8/audio-tagging-of-entity.git
   cd audio-tagging-of-entity
   ```

2. **Install dependencies:**
   - Python 3.8+
   - Install required packages:
     ```bash
     pip install numpy pandas matplotlib keras tensorflow librosa python_speech_features scikit-learn tqdm
     ```

3. **Prepare your data:**
   - Place your `.wav` files in the `Audios/` directory, organized by category.
   - Ensure audio files are in standard WAV format.
   - The project expects audio files to be named with their category (e.g., "cat_1.wav", "dog_1.wav", "human_1.wav").

4. **Run the notebooks:**
   - Open `model.ipynb` or `testL.ipynb` in Jupyter Notebook or JupyterLab.
   - Follow the cells to train or test the model.
   - The notebooks include data preprocessing, model training, and evaluation steps.

## Model Architecture

### CNN Model
- Multiple Conv2D layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification

### RNN Model
- LSTM layers for temporal feature extraction
- TimeDistributed Dense layers
- Dropout for regularization
- Final Dense layer with softmax activation

## Example Usage

- **Training:**  
  Use `model.ipynb` to:
  - Preprocess audio data
  - Extract MFCC features
  - Build and train the model
  - Save model checkpoints

- **Testing:**  
  Use `testL.ipynb` to:
  - Load trained models
  - Evaluate model performance
  - Generate predictions
  - Analyze results

## Notes

- Make sure your audio files are in standard WAV format.
- You may need to adjust file paths in the notebooks depending on your environment.
- The model configuration can be modified in `cfg.py`.
- Training parameters can be adjusted in the notebooks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Aelish Kumar

## License

This project is licensed under the MIT License - see the LICENSE file for details.