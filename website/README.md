# FaceGuard — Railway-ready Streamlit app

Image upload, webcam snapshots, real CPU inference, model selection, test-results dashboard and JSON result download. Both architectures use the same full-image preprocessing as the training project. No fake predictions or automatically downloaded ImageNet classifiers.

## 1. Add your trained model

Copy your existing checkpoint (not ImageNet weights):

```bash
cp /path/to/face_antispoof/runs/resnet18_bce/best.pt models/resnet18.pt
# Optional:
cp /path/to/face_antispoof/runs/vit_b_16_bce/best.pt models/vit_b_16.pt
```

Expected checkpoint dictionary: `state`, `threshold`, `architecture`. Older checkpoints without architecture are interpreted as ResNet18. Scores use saved validation thresholds, not 0.5. Checkpoints were not supplied with this package.

Alternatively configure `RESNET18_URL` and optionally `VIT_B_16_URL` with trusted direct HTTPS checkpoint download URLs. These are server-side environment variables, never browser inputs. Ordinary sharing-page URLs will not work: the response must be the .pt bytes. For private hosting use a signed direct URL; renew expired URLs before a fresh deployment. Do not put secrets into Git. Downloaded weights are cached on the service filesystem, which may reset on redeployment.

## 2. Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
streamlit run app.py
```

For a Linux CPU-only install, use the CPU index in the Dockerfile. Default inference uses CPU for either architecture. First load/download is excluded from displayed latency. The cache holds one model to limit memory; switching architectures reloads it.

## 3. Deploy on Railway

1. Extract this ZIP. Put its contents in a GitHub repository with Dockerfile and app.py at the repository root.
2. Choose Railway **New Project → Deploy from GitHub repo**, and select that repository. If it is inside a larger repository, set the service root directory to this app's folder.
3. Configure model access before the demo:
   - Recommended: Railway service Variables → set `RESNET18_URL` to your trusted direct checkpoint URL. Add `VIT_B_16_URL` only if needed.
   - Or include `models/resnet18.pt` in the repository deliberately using `git add -f models/resnet18.pt`. The .gitignore excludes weights by default. Check repository file-size rules first; the ViT checkpoint is too large for an ordinary GitHub file. Use URL hosting for ViT.
4. Leave the start command unset. The Dockerfile calls `start.sh`, binds `0.0.0.0`, and reads Railway's `PORT` automatically.
5. Deploy and open the service logs. The healthcheck verifies the web server, not model inference: without weights the site still loads and displays a setup message.
6. In service Settings → Networking, generate a public domain. Open the HTTPS URL and allow browser camera permissions.
7. Test one image, confirm the displayed saved threshold, then test webcam capture. Allow time for the first model download/load.

Use a service memory limit sufficient for PyTorch plus the selected model. ViT consumes considerably more memory than its weight file alone. Start with ResNet18, monitor service RAM, and increase resources if the process is killed. Concurrent requests are serialized per loaded model for predictable CPU inference. Hosting costs depend on Railway resources and usage.

Optional variables: `MODEL_DIR` (default models), `TORCH_NUM_THREADS` (default 2). For a volume, set MODEL_DIR to a folder on its mount path to retain downloaded weights across redeployments.

## Included files

- app.py — application UI and results dashboard
- inference.py — checkpoint loading and real model inference
- Dockerfile — Python 3.11 + CPU PyTorch build
- start.sh — Railway port binding
- railway.json — Docker build and Streamlit healthcheck
- requirements.txt — pinned app dependencies
- .streamlit/config.toml — light theme and upload limits

## Demo limits

- Snapshot input only; no continuous streaming.
- No face detector/cropping; blank or non-face images can receive predictions. Match the full-image training pipeline.
- Binary genuine/spoof output, no attack subtype classification.
- Spoof score is not a calibrated probability of authenticity.
- Uploads are processed on the server, not only in the browser. App code does not save user images to disk or cache them globally. Use images with consent.
- Results shown are the provided custom test-split results, not official benchmark or live-camera results.

## Verification status

Python syntax and deployment configuration were checked during packaging. Real inference, Docker build, browser rendering, and Railway deployment require your checkpoints and an installed runtime; they have not been executed in the packaging environment.

References: https://docs.railway.com/deployments/start-command and https://docs.streamlit.io/deploy/tutorials/docker
