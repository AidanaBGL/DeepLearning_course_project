# Face Anti-Spoofing Detection

**Team:** Aidana, Diana, Temirlan  
**Current stage:** Binary baselines trained and evaluated  
**Last updated:** 17 November 2026

## Task list

- [x] Set up the project code and record dependency requirements.
- [x] Download the available CelebA-Spoof archive parts and inspect the extracted images and folder labels.
- [x] Check image integrity and prepare identity-separated train, validation, and test splits.
- [ ] Implement and compare two versions: with and without face detection and cropping.
- [x] Train binary genuine/spoof baselines using ResNet18 and ViT-B/16.
- [x] Evaluate both models and record quality metrics and parameter counts.
- [ ] Complete resource measurements: inference time, checkpoint size, RAM, and GPU memory.
- [ ] Recover attack-type annotations and train a multiclass model that distinguishes genuine faces and available attack categories.
- [ ] Suggest additional datasets or architectures for further experiments.
- [ ] Complete and validate the Streamlit website with image upload and webcam capture.
- [ ] Prepare printed photographs and phone/tablet screen attacks for the live demo.

## Project overview

Face verification systems can encounter attacks such as printed photographs or faces displayed on a screen. We want to train a model that distinguishes a genuine face presentation from a spoofing attempt and demonstrate it through a Streamlit website.

During the presentation, we will show a real face to the webcam, then try printed photographs and faces displayed on a phone or tablet. The website will show the prediction, model score, and processing time. We will also test image uploads.

Our main question is: **How well can a model trained on a public dataset detect attacks captured using our own camera, lighting, and devices?**

The current binary baselines use full images. Face cropping, multiclass prediction, and live-camera evaluation are next steps. This project focuses on physical presentation attacks; detecting digitally generated deepfakes would require separate data and evaluation.

## Classes and attack types

The trained models predict two classes: **genuine** and **spoof**. They do not currently identify the type of attack.

| Category to investigate | Example | Planned use |
| --- | --- | --- |
| Genuine / bona fide | A real person in front of the camera | Training, evaluation, and live demo |
| Print attack | A face printed on paper and shown to the webcam | Main attack for the live demo |
| Display attack: still image | A face photograph shown on a phone, tablet, or monitor | Main attack for the live demo |
| Display attack: video replay | A prerecorded face video played on a screen | Additional demo if feasible |
| Other annotated attacks | Masks or other attack categories supported by the recovered annotations | Optional training and evaluation |

These are our planned evaluation groups. CelebA-Spoof provides detailed spoof-type annotations, but the current manifests only use the `live` and `spoof` folder labels. The field `attack = -1` means **unknown subtype**.

For multiclass training, we need to match the recovered images with their original attack annotations and inspect class counts. The final mapping could include genuine, print, display, and other supported categories. Still-image displays and video replays must remain grouped if the annotations do not distinguish them. The website will only show predicted subtypes after a multiclass model has been trained and evaluated.

## Face Anti-Spoofing Project Report
[ Presentation](https://canva.link/51ajxm1rksf7nk8) [Website](https://deeplearningcourseproject-production.up.railway.app/)

### Website demo

> **Screenshot placeholder:** Add a screenshot of the working Streamlit application here, showing an input image, prediction, spoof score, and processing time.

<!-- When the screenshot is available, add it to assets/website-demo.png and uncomment:
![Streamlit face anti-spoofing demo](assets/website-demo.png)
-->

## Results

### Dataset used

Because some archive parts were missing, only part of CelebA-Spoof was recovered. 

| Split | Images | People | Genuine | Spoof |
| --- | ---: | ---: | ---: | ---: |
| Training | 20,198 | 210 | 4,298 | 15,900 |
| Validation | 3,903 | 45 | 935 | 2,968 |
| Test | 5,200 | 46 | 1,072 | 4,128 |
| **Total** | **29,301** | **301** | **6,305** | **22,996** |

### Model pipeline

| Setting | Baseline configuration |
| --- | --- |
| Architectures | ResNet18 and ViT-B/16 |
| Initialization | ImageNet-pretrained weights |
| Input | Full RGB image, 224 × 224 |
| Output labels | Genuine = 0; spoof = 1 |
| Training duration | 10 epochs per model |
| Intended comparison settings | BCE loss, AdamW, batch size 16, learning rate 0.0001, seed 42 |
| Training augmentation | Horizontal flips and moderate brightness/contrast changes |
| Checkpoint selection | Lowest validation ACER |
| Decision threshold | Selected on validation for each checkpoint; fixed during test evaluation |

### Evaluation

<img width="200" height="200" alt="ResNet18_Training" src="https://github.com/user-attachments/assets/337fdbf1-aeea-4a09-a3c3-2495b0549370" />

| Model | AUC ↑ | EER ↓ | APCER ↓ | BPCER ↓ | ACER ↓ | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ResNet18** | **0.99815** | **1.44%** | **0.94%** | **2.24%** | **1.59%** | **11.18 M** |
| ViT-B/16 | 0.99421 | 3.66% | 4.14% | 2.33% | 3.24% | 85.80 M |

- **APCER:** percentage of attacks incorrectly accepted as genuine.
- **BPCER:** percentage of genuine samples incorrectly rejected as attacks.
- **ACER:** average of APCER and BPCER.
- **EER:** approximate equal-error rate across thresholds, not necessarily at the deployed threshold.
- **AUC:** ranking performance across thresholds, reported here on a 0–1 scale.

### confusion matrix

| Outcome | ResNet18 | ViT-B/16 |
| --- | ---: | ---: |
| Genuine correctly accepted | 1,048 | 1,047 |
| Genuine incorrectly rejected | 24 | 25 |
| Attacks incorrectly accepted | 39 | 171 |
| Attacks correctly rejected | 4,089 | 3,957 |

### Training time and resource measurements

| Measurement | ResNet18 | ViT-B/16 |
| --- | ---: | ---: |
| Sum of 10 training-loop durations | 80 min 47 s | 99 min 49 s |
| Mean training-loop duration per epoch | 8 min 05 s | 9 min 59 s |
| Parameter count | 11.18 M | 85.80 M |
| Checkpoint file size | Not measured | Not measured |
| Inference latency | Not measured | Not measured |
| Peak process RAM | Not measured | Not measured |
| Peak GPU memory | Not measured | Not measured |

## How to run the code

Run the following commands from the project directory. Use the same CSV manifests for both architectures.

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the recovered subset

```bash
python prepare.py \
  --root extracted/CelebA_Spoof/Data/test \
  --out data/manifests
```

### 3. Train the two baselines

```bash
python train.py --model resnet18 --loss bce --epochs 10 --batch-size 16 --lr 0.0001 --seed 42 --out runs/resnet18_bce

python train.py --model vit_b_16 --loss bce --epochs 10 --batch-size 16 --lr 0.0001 --seed 42 --out runs/vit_b_16_bce
```

### 4. Evaluate and compare

```bash
python evaluate.py --checkpoint runs/resnet18_bce/best.pt --out runs/eval_resnet18

python evaluate.py --checkpoint runs/vit_b_16_bce/best.pt --out runs/eval_vit

python compare.py runs/eval_resnet18 runs/eval_vit
```


## Resources

| Resource | Link | Use in this project |
| --- | --- | --- |
| CelebA-Spoof | [Official repository and dataset download links](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | Dataset, annotations, and usage terms |
