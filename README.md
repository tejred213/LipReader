# LipReader - Reads lips from video and predicts the spoken text.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)  ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen?style=for-the-badge&logo=opencv&logoColor=white)  ![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&logo=github)  ![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge&logo=github)   

LipNet is a deep learning-based lip-reading model that processes video sequences to predict the spoken text. This project is inspired by the original LipNet paper and showcases the application of convolutional and recurrent neural networks for sequence-to-sequence tasks.

---

## 🚀 Features
- **Video Data Processing**: Custom pipeline for loading and preprocessing video frames.
- **Hybrid Architecture**: Combines CNNs and RNNs for feature extraction and temporal modeling.
- **Sequence Alignment**: Implements Connectionist Temporal Classification (CTC) for training.
- **Real-time Predictions**: Provides inference on unseen videos with text decoding.
- **Training Optimizations**: Includes learning rate schedulers, checkpoints, and real-time evaluation.

---

## 🛠️ Technologies Used
 ![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)  
 ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)  
 ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen?style=for-the-badge&logo=opencv&logoColor=white)  
 ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=for-the-badge&logo=plotly&logoColor=white)  
 ![ImageIO](https://img.shields.io/badge/ImageIO-Processing-lightblue?style=for-the-badge)  

---

## 📖 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tejred213/LipReader.git
   cd LipReader
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets using the preconfigured script in the notebook.

---

## 🧠 Project Workflow
1. **Data Loading**:
   - Videos and alignments are preprocessed to extract frames and map corresponding text annotations.

2. **Model Design**:
   - A Sequential model combining:
     - 3D Convolutional layers for spatial feature extraction.
     - Bidirectional LSTMs for temporal modeling.
     - Dense layers for classification.

3. **Training**:
   - Implements CTC loss for sequence alignment.
   - Includes checkpoints and real-time evaluation during training.

4. **Inference**:
   - Predicts text from unseen videos.
   - Decodes sequences for evaluation.

---

## 🎥 Example Usage
### Training the Model
Run the Jupyter Notebook to train the model using the preprocessed data:
```bash
jupyter notebook LipNet.ipynb
```

### Inference on a New Video
Use the following code snippet to make predictions:
```python
from model import load_model, predict
video_path = "path_to_video.mp4"
prediction = predict(video_path)
print(f"Predicted Text: {prediction}")
```

---

## 🏆 Results
- The model achieves **real-time lip-reading predictions** with high accuracy.
- Example:
  - **Input Video**: Person saying "hello world".
  - **Real Text**: "hello world".
  - **Predicted Text**: "hello world".

---

## 📊 Model Architecture
- **Input Shape**: `(75, 46, 140, 1)` (75 frames, 46x140 resolution, grayscale)  
- **Output**: Decoded text sequence using CTC.

---
## 📜 License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
