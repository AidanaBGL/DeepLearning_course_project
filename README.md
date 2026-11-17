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

| Project material | Link / status |
| --- | --- |
| Presentation | To be added |
| Project paper / report | To be added |
| Streamlit website | To be added after testing |

### Website demo

> **Screenshot placeholder:** Add a screenshot of the working Streamlit application here, showing an input image, prediction, spoof score, and processing time.

<!-- When the screenshot is available, add it to assets/website-demo.png and uncomment:
![Streamlit face anti-spoofing demo](assets/website-demo.png)
-->

### Work completed by Aidana

- Extracted the available dataset files and inspected the `live/spoof` structure.
- Checked image integrity and excluded one truncated image.
- Prepared binary labels and separate train, validation, and test groups by person ID.
- Trained ResNet18 and ViT-B/16 for 10 epochs each.
- Selected checkpoints and thresholds using validation data.
- Evaluated both models on the held-out test split.
- Prepared training curves and compared model quality and parameter counts.

The current results support using **ResNet18 as the first model for the live demo**. The Streamlit starter code exists, but successful end-to-end webcam testing has not yet been reported.

## Results

### Dataset used

Because some archive parts were missing, only part of CelebA-Spoof was recovered. The usable files were under the original `Data/test` directory. One damaged image was excluded:

```text
Data/test/6313/spoof/512099.png
```

| Category | Valid images |
| --- | ---: |
| Genuine (`live`) | 6,305 |
| Spoof | 22,996 |
| **Total** | **29,301** |

We created a new split within this recovered subset. Approximately 70% / 15% / 15% of identities were assigned to training, validation, and testing. All images of one person remain in the same split.

| Split | Images | People | Genuine | Spoof |
| --- | ---: | ---: | ---: | ---: |
| Training | 20,198 | 210 | 4,298 | 15,900 |
| Validation | 3,903 | 45 | 935 | 2,968 |
| Test | 5,200 | 46 | 1,072 | 4,128 |
| **Total** | **29,301** | **301** | **6,305** | **22,996** |

> These results use a **partially recovered CelebA-Spoof subset with a custom identity-separated split**. They are not results on the official CelebA-Spoof evaluation protocol and should not be compared directly with published official-protocol scores.

### Model pipeline

1. Read the full RGB image.
2. Resize to **224 × 224** and apply ImageNet normalization.
3. Pass the image through **ResNet18 or ViT-B/16**.
4. Apply sigmoid to obtain a spoof score.
5. Compare the score with the validation-selected threshold to predict **genuine / spoof**.

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

Check each saved `config.json` to confirm exact run settings before final reporting. The commands below use the intended BCE configuration. Full-image processing is used consistently in training and inference; the `_BB.txt` bounding boxes are not used by this baseline.

### Selected checkpoints

Both models reached their lowest validation ACER at **epoch 4**.

| Model | Best epoch | Validation AUC | Validation EER | Validation ACER | Saved threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 4 | 0.99594 | 3.01% | 2.73% | 0.194559 |
| ViT-B/16 | 4 | 0.99032 | 3.95% | 3.79% | 0.941399 |

Training loss generally decreased, while validation metrics fluctuated and did not consistently improve after epoch 4. We therefore evaluated the best saved checkpoint rather than the last epoch.

### Held-out test performance

| Model | AUC ↑ | EER ↓ | APCER ↓ | BPCER ↓ | ACER ↓ | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ResNet18** | **0.99815** | **1.44%** | **0.94%** | **2.24%** | **1.59%** | **11.18 M** |
| ViT-B/16 | 0.99421 | 3.66% | 4.14% | 2.33% | 3.24% | 85.80 M |

- **APCER:** percentage of attacks incorrectly accepted as genuine.
- **BPCER:** percentage of genuine samples incorrectly rejected as attacks.
- **ACER:** average of APCER and BPCER.
- **EER:** approximate equal-error rate across thresholds, not necessarily at the deployed threshold.
- **AUC:** ranking performance across thresholds, reported here on a 0–1 scale.

### Errors in actual images

| Outcome | ResNet18 | ViT-B/16 |
| --- | ---: | ---: |
| Genuine correctly accepted | 1,048 | 1,047 |
| Genuine incorrectly rejected | 24 | 25 |
| Attacks incorrectly accepted | 39 | 171 |
| Attacks correctly rejected | 4,089 | 3,957 |

ResNet18 achieved lower error rates in this experiment and accepted substantially fewer attacks. It also has approximately **7.7 times fewer parameters**. This does not establish a measured inference-speed advantage.

### Training time and resource measurements

The progress bars provide training-loop durations. The totals below **exclude validation, initialization, checkpoint writing, and weight downloads**, so they are not total experiment wall-clock times.

| Measurement | ResNet18 | ViT-B/16 |
| --- | ---: | ---: |
| Sum of 10 training-loop durations | 80 min 47 s | 99 min 49 s |
| Mean training-loop duration per epoch | 8 min 05 s | 9 min 59 s |
| Parameter count | 11.18 M | 85.80 M |
| Checkpoint file size | Not measured | Not measured |
| Inference latency | Not measured | Not measured |
| Peak process RAM | Not measured | Not measured |
| Peak GPU memory | Not measured | Not measured |

Record hardware, input size, batch size, and device alongside future speed/memory measurements. Parameter count is not the same as runtime memory usage. Timing comparisons also require confirmation that both runs used comparable hardware and workloads.

## How to run the code

Run the following commands from the project directory. Use the same CSV manifests for both architectures.

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build if required by the training machine. Save the working environment:

```bash
pip freeze > environment-lock.txt
```

### 2. Prepare the recovered subset

Use the **updated folder-based `prepare.py`**, not the original version requiring JSON files:

```bash
python prepare.py \
  --root extracted/CelebA_Spoof/Data/test \
  --out data/manifests
```

The script checks image decoding, skips damaged files, assigns labels from folder names, and splits identities into separate groups. It creates `train.csv`, `val.csv`, `test.csv`, and `skipped.csv`.

Preserve these manifests for all comparisons. They contain absolute image paths, so collaborators must use the same data location or update the paths without changing split membership. The original starter ZIP contained the JSON-based preparation script: replace it with the updated version used for this experiment.

### 3. Train the two baselines

```bash
python train.py --model resnet18 --loss bce --epochs 10 --batch-size 16 --lr 0.0001 --seed 42 --out runs/resnet18_bce

python train.py --model vit_b_16 --loss bce --epochs 10 --batch-size 16 --lr 0.0001 --seed 42 --out runs/vit_b_16_bce
```

Each run saves its selected `best.pt`, training configuration, and epoch history. ImageNet initialization weights are downloaded on first use if not already cached. Use a new output folder for additional experiments to preserve these runs.

### 4. Evaluate and compare

```bash
python evaluate.py --checkpoint runs/resnet18_bce/best.pt --out runs/eval_resnet18

python evaluate.py --checkpoint runs/vit_b_16_bce/best.pt --out runs/eval_vit

python compare.py runs/eval_resnet18 runs/eval_vit
```

Outputs include `metrics.json`, per-image `predictions.csv`, and `runs/comparison.csv`. Test evaluation is already complete for the two reported runs. Future model selection and tuning should use validation, not repeated optimization against these test results.

The current `attack_APCER_percent['-1']` is pooled over spoof images with unknown subtypes. It does not measure a separate attack category.

### 5. Launch Streamlit

```bash
streamlit run app.py
```

Enter `runs/resnet18_bce/best.pt` in the sidebar to use the current demo candidate. The app reads the architecture and decision threshold from the checkpoint.

The starter application supports uploads and webcam snapshots. Continuous video streaming and automatic face detection are not implemented. Its binary output should label both printed-photo and screen attacks as **spoof**, without claiming to know the attack subtype.

## Next experiments and live demo

### Full image vs. face crop

Compare a full-image version against a face-detection-and-cropping version using the same identity splits. Apply consistent cropping during training, validation, test, and website inference. Do not add cropping only at deployment and assume the existing checkpoint will behave the same way.

### Multiclass model

Recover the original subtype annotations, check how many valid images exist per category, define the class mapping, and train a multiclass head. Report macro F1, per-class recall, and a confusion matrix. This is planned work; the current checkpoints remain binary.

### Webcam demonstration

Prepare consent-based photographs, printed copies, a webcam, and a phone/tablet. Demonstrate genuine faces, paper prints, and screen presentations. Record true attack type, prediction, score, and processing time. Report local-device performance separately from the dataset test, including failed detections.

## Resources

| Resource | Link | Use in this project |
| --- | --- | --- |
| CelebA-Spoof | [Official repository and dataset download links](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | Dataset, annotations, and usage terms |
| PyTorch | [Installation instructions](https://pytorch.org/get-started/locally/) | Training environment |
| Torchvision | [Model documentation](https://docs.pytorch.org/vision/stable/models.html) | ResNet18 and ViT implementations |
| Streamlit | [Documentation](https://docs.streamlit.io/) | Website prototype |
| iBeta | [Biometric testing](https://www.ibeta.com/biometric-testing/) | Reference for future evaluation design |
| NIST | [Presentation attack detection evaluation](https://pages.nist.gov/frvt/html/frvt_pad.html) | Reference for future evaluation design |

The project has not undergone independent iBeta or NIST testing. Follow the dataset's research-use and redistribution conditions, and do not publish dataset images or personal webcam captures in the repository.
