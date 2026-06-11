# 🩺 MedReCo / MedReCo-VLM

> Official implementation of **MedReCo** and **MedReCo-VLM** — a vision-language framework for **comparative reasoning in radiology**.

**MedReCo** supports entity-aware medical image retrieval, while **MedReCo-VLM** extends it to comparative visual question answering across medical images.

---

## 📖 Overview

Radiological diagnosis often relies on comparing a current study with prior exams or clinically similar reference cases. MedReCo is purpose-built for this setting and supports:

* 🔍 **Controllable Medical Image Retrieval** — retrieve clinically analogous cases conditioned on anatomical structures, abnormal findings, or pathological conditions.
* 🎯 **Reranking** — refine coarse retrieval results with token-level query–candidate comparison.
* 💬 **Comparative VQA** — answer questions that compare two medical images.

🤗 **Pretrained weights** are available on Hugging Face:

```text
https://huggingface.co/timeseed/MedReCoVLM
```

> ⚠️ **Note:** This release currently provides **train code**, **inference code** and **complete pretrained weights**. Full training and evaluation scripts will be released together with the organized dataset.

---

## ⚙️ Installation

```bash
conda create -n MedReCo python=3.11.12
conda activate MedReCo
pip install -r requirements.txt

cd transformer_maskgit
pip install -e .

cd ../CT_CLIP
pip install -e .

cd ..
```

---

## 📂 Repository Structure

```text
MedReVLM/
├── configs/                      # Model and processor configs
├── inference/                    # Inference scripts and examples
│   ├── examples/                 # Demo images and JSON manifests
│   ├── retrieval_inference.py    # Coarse embedding + reranker inference
│   ├── retrieval_eval_dataset.py # Image loading and preprocessing
│   └── vqa_inferece.py           # Comparative VQA inference
├── src/                          # MedReCo-VLM model, dataset, trainer code
├── scripts/                      # Retrieval training/evaluation utilities
├── CT_CLIP/                      # CT-CLIP based retrieval components
├── transformer_maskgit/          # CTViT / visual encoder components
└── requirements.txt
```

---

## ⬇️ Download Weights

```bash
pip install -U huggingface_hub

huggingface-cli download timeseed/MedReCoVLM \
  --local-dir checkpoints/MedReCoVLM \
  --local-dir-use-symlinks False
```

> 💡 After downloading, remember to update the checkpoint paths in the inference scripts.

---

## 🔍 Retrieval Inference

`inference/retrieval_inference.py` performs:

1. **Coarse embedding generation**
2. **Candidate reranking**
3. **Final score fusion**
4. **Saving normalized embeddings** as `.npy` files

### 🛠️ Configuration

Edit the configuration block:

```python
CHECKPOINT = "checkpoints/MedReCoVLM/<retrieval_checkpoint>.pt"
TEXT_ENCODER_NAME = "FremyCompany/BioLORD-2023"
CONDITION_INDEX_JSON = "configs/all_condition_index.json"
DEMO_MANIFEST = "inference/examples/demo_cases.json"
ALPHA = 0.7
```

### 📄 Example Manifest

```json
{
  "modality": "2D-CXR",
  "dataset_key": "MIMIC-CXR",
  "condition": "Pulmonary Parenchyma",
  "cases": [
    {"id": "query_case_id", "image": "inference/examples/query.jpg"},
    {"id": "candidate_case_id_1", "image": "inference/examples/candidate_1.jpg"},
    {"id": "candidate_case_id_2", "image": "inference/examples/candidate_2.jpg"}
  ]
}
```

### ▶️ Run

Run from the repository root:

```bash
mkdir -p inference/outputs
python inference/retrieval_inference.py
```

### 🧬 Supported Modalities

| Modality | Description |
| :--- | :--- |
| `2D-CXR` | 2D Chest X-Ray |
| `3D-CT` | 3D Computed Tomography |
| `3D-Brain-MRI` | 3D Brain MRI |
| `2D-Ultrasound` | 2D Ultrasound |

---

## 💬 Comparative VQA Inference

`inference/vqa_inferece.py` performs single-case comparative VQA.

### 🛠️ Configuration

```python
MODEL_PATH = "checkpoints/MedReCoVLM/<vqa_model_dir>"
PROCESSOR_PATH = "configs"
```

### 📄 Example Input

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "inference/examples/vqa_image1.jpg"},
            {"type": "image", "image": "inference/examples/vqa_image2.jpg"},
            {
                "type": "text",
                "text": "Comparing the two cases, how does the endotracheal tube tip's position relative to the carina differ between Case A and Case B?"
            }
        ]
    }
]
```

### ▶️ Run

```bash
python inference/vqa_inferece.py
```

---

## 🚀 Training and Evaluation

The repository contains model definitions, dataset utilities, trainer code, and training-related modules. The **complete reproducible training and evaluation scripts** will be released after the dataset is organized and uploaded. Stay tuned! 🔔

---

## 🙏 Acknowledgements

This codebase is built upon the following excellent projects:

* 🔗 [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)
* 🔗 [ibrahimethemhamamci/CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP)

We sincerely thank the authors for their high-quality open-source work.

---

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@article{zhang2026vision,
  title={A Vision-language Framework for Comparative Reasoning in Radiology},
  author={Zhang, Tengfei and Zhao, Ziheng and Dai, Lisong and Zhang, Xiaoman and Qiu, Pengcheng and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  journal={arXiv preprint arXiv:2606.06407},
  year={2026}
}
```

---

## ⚠️ Disclaimer

This project is intended for **research use only** and is **not a medical device**.
