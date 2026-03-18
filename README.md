<h1 align="center">X-Raydar CV</h1>

<p align="center">
  <a href="https://x-raydar.info"><img src="https://www.x-raydar.info/img/logos/logo-online.png" alt="X-Raydar" /></a>
</p>

<p align="center">
  <a href="https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00218-2/fulltext">Paper</a> &middot;
  <a href="https://huggingface.co/dnamodel/xraydar-cv">Model Weights</a> &middot;
  <a href="https://x-raydar.info">Website</a>
</p>

Computer vision component of [X-Raydar](https://x-raydar.info), from ["Development and validation of open-source deep neural networks for comprehensive chest x-ray reading"](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00218-2/fulltext) (Cid, Macpherson et al., *The Lancet Digital Health*, 2024).

A multi-scale Inception v3 ensemble (XNet38MS) that detects **37 radiological findings** in chest X-rays. Three models at resolutions 299, 512, and 1024 pixels are averaged at inference time. Trained on over 2.5 million studies from six NHS hospitals.

> **NOTE: This is not for clinical use.**

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights

```python
from huggingface_hub import hf_hub_download
import shutil, os

for size in [299, 512, 1024]:
    dest = f"src/model_20210820_XNet38MS/model_weights/direct_multi93_is{size}_Rv10_pre00_imagenet"
    os.makedirs(dest, exist_ok=True)
    for f in ["model_best.pth.tar", "model_TranslatorCVLogitsToUrgency_fcs.pth.tar"]:
        path = hf_hub_download("dnamodel/xraydar-cv", f"cv/is{size}/{f}")
        shutil.copy(path, os.path.join(dest, f))
```

### 3. Run inference

```python
import pydicom
import utils.dicom_utils as dicom_utils
import model_20210820_XNet38MS.predict as predict

# Build the multi-scale model (loads 3 resolutions)
model = predict.build_model()

# Load and preprocess a DICOM file
dicom = pydicom.dcmread("demo_data/04f72062c19d9cd7a55519708aa2cc58b5e52b52")
image = dicom_utils.img_clean(dicom)

# Predict 37 radiological findings
report = predict.main(image, model)
print(report["AI_prediction"])
```

Demo DICOM files are provided in `demo_data/`.

## Project Structure

```
src/
├── model_20210820_XNet38MS/
│   ├── predict.py              # Inference (build_model, prepare_data, test)
│   ├── XNet38_urg.py           # XNet38 + urgency head wrapper
│   ├── wt_inception.py         # Modified Inception v3 (single-channel)
│   └── model_weights/          # Place downloaded weights here
├── utils/
│   ├── dicom_utils.py          # DICOM loading and preprocessing
│   ├── image_utils.py          # Image resizing and normalization
│   └── report_utils.py         # Report formatting
└── inference_script.ipynb      # Example notebook
```

## Requirements

- Python 3.8+
- PyTorch
- pydicom
- Pillow, NumPy, scikit-image

See `requirements.txt` for pinned versions.

## Related

- **NLP model** (radiology report classifier): [x-raydar-nlp](https://github.com/x-raydar/x-raydar-nlp) &middot; [HuggingFace](https://huggingface.co/dnamodel/xraydar-nlp)

## Citation

```bibtex
@article{cid2024development,
  title={Development and validation of open-source deep neural networks for
         comprehensive chest x-ray reading: a retrospective, multicentre study},
  author={Cid, Yan Digilov and Macpherson, Matt and Gervais-Andre, Luc and
          Zhu, Yinghui and Franco, Guillermo and Santeramo, Ruggiero and
          Mudali, Divya and Wood, Orlando and Montague, Eoin and Wei, Jiefei and
          others},
  journal={The Lancet Digital Health},
  volume={6}, number={1}, pages={e44--e57},
  year={2024}, publisher={Elsevier},
  doi={10.1016/S2589-7500(23)00218-2}
}
```

## License

Academic research and non-commercial evaluation only. See [LICENSE](LICENSE) for full terms.

## Contact

Giovanni Montana — [g.montana@warwick.ac.uk](mailto:g.montana@warwick.ac.uk)

Commercial licensing — Warwick Ventures — [ventures@warwick.ac.uk](mailto:ventures@warwick.ac.uk)
