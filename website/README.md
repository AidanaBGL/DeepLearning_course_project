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
