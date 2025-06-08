# audio-tagging-of-entity

A machine learning project for classifying and tagging audio files (such as cat, dog, and human sounds) using deep learning models.

## Project Structure

- `Audios/` — Contains audio files organized by category (cat, dog, human).
- `models/` — Stores trained model files.
- `model.ipynb` / `testL.ipynb` — Jupyter notebooks for training and testing models.
- `audio_labels_with.csv`, `audio_labels.csv`, `audio_labels_length.csv` — CSV files with audio metadata and labels.
- `cfg.py` — Configuration file for model parameters.
- `count_entity.py` — (Empty or utility script, can be used for entity counting).
- `old_data/` — Contains older datasets and scripts.

## Features

- Audio classification for multiple entities (cat, dog, human).
- Deep learning models: Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN/LSTM).
- Data augmentation and preprocessing utilities.
- Jupyter notebooks for experimentation and visualization.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aelishkumar8/audio-tagging-of-entity.git
   cd audio-tagging-of-entity
   ```

2. **Install dependencies:**
   - Python 3.8+
   - Install required packages (example, adjust as needed):
     ```bash
     pip install numpy pandas matplotlib keras tensorflow librosa python_speech_features scikit-learn tqdm
     ```

3. **Prepare your data:**
   - Place your `.wav` files in the `Audios/` directory, organized by category.

4. **Run the notebooks:**
   - Open `model.ipynb` or `testL.ipynb` in Jupyter Notebook or JupyterLab.
   - Follow the cells to train or test the model.

## Example Usage

- **Training:**  
  Use `model.ipynb` to preprocess data, build features, and train a model.
- **Testing:**  
  Use `testL.ipynb` to evaluate model performance or generate audio metadata.

## Notes

- Make sure your audio files are in standard WAV format.
- You may need to adjust file paths in the notebooks depending on your environment.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

[Specify your license here]