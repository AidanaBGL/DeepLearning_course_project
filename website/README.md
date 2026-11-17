# Face Antispoof — built-in ResNet18 settings

Replace app.py and inference.py in your existing Lab UI deployment and redeploy.
ResNet18 no longer reads local JSON or RESNET18_META_URL. Old malformed JSON files are ignored.
Keep RESNET18_ONNX_URL pointing to your exported model, or models/resnet18.onnx locally.
The threshold, preprocessing identifier and SHA256 from your supplied export are embedded in inference.py.
The model must match that export. For a new export update the built-in settings together.
ViT behavior is unchanged: it needs ONNX and matching JSON because its export checksum was not supplied.
No weights are included. A real hosted inference has not been run here.
