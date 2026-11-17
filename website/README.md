# Adding the LCC-FASD models (Diana's Colab runs)

The app lists every model in `MODELS` (inference.py) that has an `<id>.onnx` + `<id>.json` pair in `models/` or `<ID>_ONNX_URL` + `<ID>_META_URL` variables. Unconfigured models are hidden. LCC-FASD ids: `lcc_mobilenet_v3_large`, `lcc_efficientnet_b0`, `lcc_resnet18`, `lcc_deit_tiny`, `lcc_custom_cnn`.

1. Copy `export_onnx_lcc.py` to `MyDrive/FaceAntiSpoofing/` on Google Drive.
2. In the training notebook, run sections 1–2 (mount Drive, unpack the dataset), then in a new cell:
   ```
   !pip -q install onnx onnxruntime onnxscript
   !python /content/drive/MyDrive/FaceAntiSpoofing/export_onnx_lcc.py --runs /content/drive/MyDrive/FaceAntiSpoofing/runs_lcc_fasd --images /content/lcc_fasd_data/LCC_FASD_development --out /content/drive/MyDrive/FaceAntiSpoofing/onnx_lcc
   ```
   It checks ONNX vs PyTorch scores on 64 development images and refuses to write a model that disagrees.
3. Download `onnx_lcc/` from Drive and put the files into `website/models/` (all are under 50 MB), commit and push; Railway rebuilds. Alternatively upload them to Hugging Face and set e.g. `LCC_MOBILENET_V3_LARGE_ONNX_URL` / `LCC_MOBILENET_V3_LARGE_META_URL` to the `/resolve/main/...` links. Google Drive share links do not work as download URLs.

---

# Reference-layout update

Compact laboratory layout based on the supplied reference: input panel left, result panel right, suggested test cards and testing references below. Milk-beige background and orange submit button retained from the requested palette. Top links remain configurable through GITHUB_URL, HUGGINGFACE_URL, PRESENTATION_URL and PAPER_URL.

Replace app.py to update an existing cream/logs ONNX deployment. If updating an older deployment, replace the full app files while retaining your models and hosting variables. Rebuild after updating dependencies.

This is an actual inference interface, not a reproduction of the illustrative screenshot's result. No face bounding box is drawn because no detector is implemented. Browser-selected webcam snapshots remain supported; a custom camera-device selector and continuous-stream indicator are not implemented. The percentage is the actual sigmoid spoof score and not calibrated confidence. Metadata/hash errors still block inference and remain visible in the processing log; this redesign does not fix mismatched model files.

Checked: Python syntax and configuration parsing. Not checked here: live browser layout and inference with your weights.

---

# FACE ANTISPOOF — cream UI + processing logs

## Update your deployed site

Replace app.py, inference.py, requirements.txt and .streamlit/config.toml with this package, then rebuild the deployment. Keep your existing ONNX/JSON model files and model environment variables. Do not use the old PyTorch app.py or inference.py.

The interface has a warm milk-beige background, FACE ANTISPOOF branding at the top left, project links at the top right, an image/camera input and model dropdown on the left, an orange Отправить button below the input, and the result on the right. No sidebar navigation.

Set these four optional environment variables to your REAL project URLs:

```text
GITHUB_URL=https://github.com/your-name/your-project
HUGGINGFACE_URL=https://huggingface.co/your-name/your-model-repo
PRESENTATION_URL=https://your-presentation-link
PAPER_URL=https://your-paper-link
```

Missing navigation links are shown as disabled gray text; the package does not invent links.

## What the logs show

After Отправить, a live status panel shows: request received → model metadata → download progress in 16 MiB increments → hash verification → ONNX initialization → preprocessing → inference → completion. Cached model loads skip download/init stages. The expandable Processing log preserves the last request's progress in that browser session. Results persist across normal reruns but reset when the image, input source or model changes.

Railway/hosting logs contain UTC-style server timestamps (host timezone), request ID, selected model, stages and exception class. The request ID is also displayed in the result/error. No image bytes, image names, signed URLs or authentication tokens are intentionally logged. Browser messages report specific causes such as HTTP errors, invalid JSON, hash mismatch and permissions instead of silently swallowing errors. Unexpected failures show the exception class and last completed stage; no full traceback with potentially sensitive URLs is displayed.

If the process is killed for insufficient memory, Python cannot catch that termination. Look at host restart/OOM events. If downloading stalls, connection reads time out after 120 seconds; this is not a strict total transfer deadline.

## Score vs confidence

The prominent percentage is the real sigmoid spoof score, not a fabricated confidence estimate. The saved validation threshold is shown separately. Because thresholds differ from 50%, a genuine prediction can have a spoof score greater than 50% (e.g. ViT). Calibrated confidence is not implemented.

## Dependency fixes

PyArrow is pinned to 18.0.0 for NumPy 1.26.4 compatibility. Streamlit file watching is disabled. Docker installs only ONNX runtime dependencies, not torch. A rebuild is required after editing requirements.

## Validation

Python syntax, TOML/JSON parsing, callback wiring and existing preprocessing smoke tests were checked locally. Browser layout and actual model inference were not executed here: no deployed service or trained model files were supplied. Review the deployment after replacing files.

---

# FaceGuard — ONNX Runtime edition

Railway-ready Streamlit demo for the existing ResNet18 and ViT-B/16 binary checkpoints. Deployment uses ONNX Runtime CPU, NumPy and Pillow — no PyTorch or torchvision. Export runs separately in your training environment.

## 1. Export on the training machine

Copy export_onnx.py, preprocess.py, requirements.txt and requirements-export.txt to a working folder, or use this whole folder. Activate your environment with torch and torchvision already installed:

```bash
pip install -r requirements-export.txt
python export_onnx.py --checkpoint /path/to/runs/resnet18_bce/best.pt --csv /path/to/data/manifests/val.csv --out models
python export_onnx.py --checkpoint /path/to/runs/vit_b_16_bce/best.pt --csv /path/to/data/manifests/val.csv --out models
```

CSV paths must point to readable images on this machine. By default, 64 validation images are selected with seed 42. To verify the complete validation set use `--samples 3903` (slower). Use validation rather than test for export troubleshooting.

If a separate export environment is needed, Python 3.11 with torch==2.5.1 and torchvision==0.20.1 is the reference dependency pair used in the previous Docker setup. Do not replace your training environment unnecessarily. This script explicitly selects the legacy ONNX exporter (`dynamo=False`, opset 17) and disables the fused MHA fast path for ViT compatibility.

Outputs:

```text
models/
  resnet18.onnx
  resnet18.json
  vit_b_16.onnx
  vit_b_16.json
```

Each metadata JSON contains the architecture, ORIGINAL checkpoint threshold, preprocessing version, model SHA256 and parity-check results. Do not mix JSON and ONNX files from different exports. Hash mismatch stops inference.

The exporter validates ONNX structure, compares Pillow/NumPy preprocessing against torchvision, compares PyTorch and ONNX sigmoid scores (absolute tolerance 1e-4), and checks predictions at the saved threshold. It only publishes the model pair after passing these checks. Passing 64 samples is a sample check, not proof of equivalence on all inputs. Exported graphs are float32 with fixed input [1,3,224,224] and logits output. No quantization is applied.

## 2. Run the website locally

In a clean deployment environment (no PyTorch needed):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Keep the exported ONNX/JSON pairs in models/. Only configured models are listed. Upload an image or capture a webcam snapshot. The app uses the same full-frame resize and ImageNet normalization as the training pipeline. Threshold is read from metadata; it is not reset to 0.5.

## 3. Deploy on Railway

1. Put these application files in a GitHub repository. Dockerfile must be at the configured Railway service root.
2. Provide the exported model pair using ONE approach:
   - Local files: include models/resnet18.onnx and models/resnet18.json in the deployed source. ONNX files are ignored by Git by default; deliberately add the ResNet file using `git add -f models/resnet18.onnx` if allowed by repository size rules. JSON is not ignored. ViT weights generally exceed ordinary GitHub per-file limits; use URLs.
   - Remote files: set Railway variables `RESNET18_ONNX_URL` and `RESNET18_META_URL` to direct HTTPS download URLs for the matching pair. Optionally add `VIT_B_16_ONNX_URL` and `VIT_B_16_META_URL`. Only use files you control/trust. Ordinary sharing-page links do not work. Private signed URLs must remain valid on fresh deployments.
3. Choose Deploy from GitHub repo. Leave the start command unset: Dockerfile calls start.sh, binding 0.0.0.0 and Railway's PORT.
4. Generate a public HTTPS domain in service networking settings and permit webcam access in your browser.
5. Run a real prediction after deployment. The web-server healthcheck does not validate model availability or inference.

Optional variables: `ORT_NUM_THREADS=2`, `MODEL_DIR=models`. For persistent weights, mount a volume and point MODEL_DIR to a folder on it. Without persistence, remote models are downloaded again after filesystem resets. Start with ResNet18 and monitor Railway RAM before enabling ViT.

The new ONNX edition does not use the previous RESNET18_URL/VIT_B_16_URL variables or .pt files. Use the four ONNX/META variable names above.

## 4. Benchmark CPU inference

Run each model in a fresh process on the same target machine:

```bash
python benchmark.py --model resnet18 --image /path/to/example.png --runs 100 --out resnet18_benchmark.json
python benchmark.py --model vit_b_16 --image /path/to/example.png --runs 100 --out vit_b_16_benchmark.json
```

The script records mean/p95 inference time, preprocessing + inference time, loading time, and process RSS sampled every 10 ms. RSS is the whole process footprint, not model-only RAM; sampled peak can miss shorter spikes. Loading can include downloading if the weights were absent. Benchmarks exclude browser/network transfer and image decoding; image loading occurs before the timed loops. Warm-up: 10 calls; batch size: 1. No GPU/VRAM is used.

For a PyTorch vs ONNX speed claim, also measure the original PyTorch model on identical hardware, thread counts, images and warm-up. The included benchmark only measures the ONNX deployment path.

## What changed

- ONNX Runtime CPU inference replaces PyTorch in the website.
- Runtime graph optimizations are enabled with ORT_ENABLE_ALL on the deployment CPU.
- Fixed batch 1 and 224x224 shape match the snapshot demo.
- Preprocessing is implemented with Pillow/NumPy and checked against torchvision during export.
- A one-entry Streamlit model cache limits retained sessions; switching models reloads the session.
- Saved thresholds, float32 precision and the existing binary label convention are preserved.

No hardware-specific pre-optimized graph is shipped; optimization runs when ONNX Runtime creates the session. This avoids assuming export-machine hardware matches Railway. FP32 ONNX conversion need not reduce the weight file size or improve latency. INT8 and FP16 are not enabled: they need separate compatibility, accuracy and performance validation.

## App behavior and limits

- Binary genuine/spoof predictions; no attack subtype classifier.
- Full frame, no face detector; non-face inputs can still receive a prediction.
- Webcam snapshots, not continuous video.
- Spoof scores are not calibrated authenticity probabilities.
- Images are sent to the server and processed in memory; application code does not save uploads to disk.
- Results tab contains the previously reported custom test results from PyTorch. These are not newly measured ONNX metrics. Re-evaluate the exported model before reporting ONNX test performance.

## Verification status

Package syntax/configuration and NumPy preprocessing smoke tests were checked. Trained .pt files were not provided, and PyTorch/ONNX Runtime are not installed in the packaging environment. Actual export, parity, latency, Docker build, and Railway inference remain to be run on your machine. The scripts perform those export checks when you run them; no fabricated benchmark results are included.

## Files

- app.py: Streamlit interface and previously reported result tables
- inference.py: secure-by-configuration model retrieval, hash validation, ONNX Runtime session and prediction
- preprocess.py: shared Pillow/NumPy preprocessing and stable sigmoid
- export_onnx.py: checkpoint-to-ONNX export and parity checks (training environment only)
- benchmark.py: CPU latency and sampled process memory measurements
- requirements.txt: Railway/runtime dependencies only
- requirements-export.txt: additional export dependencies; torch/torchvision supplied by training environment
- Dockerfile, railway.json, start.sh, .streamlit/config.toml: deployment configuration

References: https://onnxruntime.ai/docs/performance/graph-optimizations and https://docs.railway.com/deployments/start-command
