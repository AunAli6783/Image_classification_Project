# Car Detection using YOLOv5

This project uses **YOLOv5** with transfer learning on the **Stanford Cars Dataset** to detect and classify car images.

---

## 🚀 Project Structure
''' project/
├── data/
│ └── images/ ← Place all car images here (train/test)
├── models/ ← YOLOv5 models and checkpoints
├── notebooks/ ← Jupyter notebooks for training and inference
├── utils/ ← Helper functions and scripts
├── requirements.txt ← List of Python dependencies
└── README.md ← Project overview and setup instructions '''

---

## 📦 Dataset Setup

> **Note:** The actual dataset is not included in this repository due to size constraints.

To use this project, **you must manually download and place the dataset** in the correct folder:

### 📥 Download Stanford Cars Dataset

- **URL:** [http://ai.stanford.edu/~jkrause/cars/car_dataset.html](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

### 📂 Expected Structure

After downloading:

data/
└── images/
├── train/
│ └── *.jpg
└── test/
└── *.jpg


If this folder doesn't exist, create it manually:

```bash
mkdir -p data/images/train
mkdir -p data/images/test

Clone this repository:


git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install dependencies:

python -m pip install -r requirements.txt
Train the model:

python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt

🧪 Inference
To test your model on a sample image:

python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source data/images/test/

🧾 Notes
Images and datasets are not tracked in Git for performance and privacy reasons.

Ensure you have data.yaml properly configured to point to your dataset.

🙋‍♂️ Author
RAJA AUN ALI
rajaaunalikhan@gmail.com
