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

# Face Antispoof — built-in ResNet18 settings

Replace app.py and inference.py in your existing Lab UI deployment and redeploy.
ResNet18 no longer reads local JSON or RESNET18_META_URL. Old malformed JSON files are ignored.
Keep RESNET18_ONNX_URL pointing to your exported model, or models/resnet18.onnx locally.
The threshold, preprocessing identifier and SHA256 from your supplied export are embedded in inference.py.
The model must match that export. For a new export update the built-in settings together.
ViT behavior is unchanged: it needs ONNX and matching JSON because its export checksum was not supplied.
No weights are included. A real hosted inference has not been run here.
