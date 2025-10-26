---
# ⚡ Voltix — YOLO Multi-Class Object Detection (Hackathon Project)

This repository is developed by **Team Voltix** for the **Hack Of Thrones**.
It implements a YOLO-based object detection model trained on the **Falcon Duality AI dataset** to identify seven safety-related objects.

The model is designed to detect **7 safety-related objects** from images and videos.

### [**Prototype Demo Link**](https://youtu.be/saBL5PV4_QM) 
---

## 🧠 Dataset Overview

**Dataset Name:** Falcon Duality AI  
**Classes (`nc: 7`):**

```
['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
```

You can get the dataset for training as well as testing dataset on [**Falcon Duality AI**](https://falcon.duality.ai/secure/documentation/7-class-hackathon&utm_source=hackathon&utm_medium=instructions&utm_campaign=hackofthrones)

---

## ⚙️ Environment Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Yousuf-177/Voltix.git
cd Voltix
```
### 2️⃣ Navigate to model Directory



```bash
cd model
```


### 3️⃣Create a Virtual Environment (Recommended)

```bash
python -m venv yolovenv
source yolovenv/bin/activate      # On Linux/Mac
yolovenv\Scripts\activate         # On Windows
```

### 4️⃣  Install Dependencies

```bash
pip install -r requirements.txt
```
###  5️⃣ Navigate to Assets Sub-Directory


```bash
cd assets
```

---


## YOLO Detection Script: `yolo_detect.py`

This script performs inference using a trained YOLO model on images, folders of images, or video files. It also saves annotated images and bounding box labels for further analysis.

### **Usage**

```bash
python yolo_detect.py --model <path_to_model> --source <source_path> [--thresh <confidence>] [--resolution <WxH>] [--record]
````

### **Arguments**

| Argument       | Type   | Required | Default | Description                                                                                                                                                      |
| -------------- | ------ | -------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`      | string | Yes      | None    | Path to the YOLO model file (e.g., `runs/detect/train/exp/weights/best.pt`). If not provided, defaults to `yolov8s.pt` in the script directory.                  |
| `--source`     | string | Yes      | None    | Source for inference. Can be: <br>• Single image file (e.g., `test.jpg`) <br>• Folder of images (e.g., `images/`) <br>• Video file (e.g., `test.mp4`)            |
| `--thresh`     | float  | No       | 0.5     | Minimum confidence threshold for displaying predicted objects. Values range from 0.0 to 1.0. Example: `--thresh 0.4`                                             |
| `--resolution` | string | No       | None    | Resolution for displaying annotated images/videos in `WxH` format. If not specified, the script will use the source resolution. Example: `--resolution 1280x720` |
| `--record`     | flag   | No       | False   | Record a video of the predictions. If used, `--resolution` must also be specified. Output is saved as `demo1.avi` in the script directory.                       |

### **Outputs**

1. **Annotated images:** Saved under `predictions/images/`
2. **Bounding box labels:** Saved under `predictions/labels/` in YOLO format `[class_id x_center y_center width height]`
3. **Validation Report:** The validation report of every testing will saved at `runs/detect/val{}`
3. **Optional video recording:** If `--record` is used, saved as `demo1.avi`

### **Example Commands**

**Single image inference:**

```bash
python yolo_detect.py --model runs/detect/train/exp/weights/best.pt --source test1.jpg
```

**Folder of images:**

```bash
python yolo_detect.py --model runs/detect/train/exp/weights/best.pt --source images/ --thresh 0.4
```

**Video inference with recording:**

```bash
python yolo_detect.py --model runs/detect/train/exp/weights/best.pt --source test_video.mp4 --resolution 1280x720 --record
```



---

## 🧩 Training the Model

To start model training:

```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt --epochs 100 --mosaic 0.50 --optimizer AdamW --momentum 0.9
```

You can modify `--epochs` `--mosaic` `--optimizer` `--momentum` as per your choices

This will automatically create the following directory:

```
runs/detect/train/
 ├── weights/
 │   ├── best.pt
 │   └── last.pt
 ├── results.png
 ├── confusion_matrix.png
 └── metrics.csv
```

---

## 🧪 Testing / Evaluating the Model

### Run Evaluation on Validation Set

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=dataset/data.yaml
```

---

## 📊 Model Performance

| Metric                   | Value | Notes                                                 |
| :----------------------- | :---: | :---------------------------------------------------- |
| **mAP@0.5**              | 0.794 | (Mean Average Precision at IoU threshold 0.5)         |
| **Precision**            | 0.993 | (Proportion of correct positive predictions)          |
| **Recall**               | 0.81  | (Proportion of actual positives correctly identified) |
| **F1-Score**             |0.80 at 0.356| Harmonic mean of precision and recall        |

---

## 🔁 Reproducing Final Results

To reproduce the same model results:

1. Clone the repo and set up the environment as per [Environment Setup](#️-environment-setup).
2. Use the same dataset structure (`data.yaml`, train/val/test splits).
3. Train using the same configuration command.
4. Evaluate using the same validation command.
5. The results (metrics, weights, and logs) will be stored under `runs/detect/train/`.

---

## 🖼️ Expected Outputs

### During Training:

- `results.png` → shows training & validation loss, precision, recall, and mAP curves
- `weights/best.pt` → best model based on validation performance
- `confusion_matrix.png` → visual summary of class-wise predictions

### During Testing:

- Output images/videos with bounding boxes, class labels, and confidence scores
- Example:
  ```
  detections/
   ├── img1_pred.jpg
   ├── img2_pred.jpg
  ```

Interpret the results as follows:

- **Bounding Box Color:** Represents detected class
- **Confidence Score:** Model’s certainty about the detection
- **Low Confidence (<0.4)** → may indicate false positives or ambiguous cases

---

## 🧾 Notes

- Modify hyperparameters in `config.yaml` or directly in the training command for optimization.
- Ensure the dataset is correctly annotated in YOLO format (one `.txt` file per image).
- Use GPU for faster training and inference (`torch.cuda.is_available()`).

---

## 📁 Repository Structure

```
📦 Voltix
 ┣  📂 models
 |   ┣ 📂 assets
 |   ┣ 📂 runs/detecs
 |   ┣ 📜 predict.py
 |   ┣ 📜 recquirement.txt
 |   ┣ 📜 train.py
 |   ┣ 📜 yolo_params.yaml
 |   ┗ 📜 yolov8s.pt
 ┣  📜 README.md
 ┗  📜 Hackathon Report — Team Voltix
 
```

---
