# MedInjection-FR
Here’s a plug-and-play `README.md` for your GitHub repo. It links to the three Hugging Face datasets and the fine-tuned models, and it includes quickstart code, stats, evaluation notes, citation, and license.

---

# MedInjection-FR

A French biomedical **instruction dataset** and **model suite** for studying how data provenance (native, synthetic, translated) impacts instruction-tuning of LLMs.

* **Total size:** 570,154 instruction–response pairs
* **Components:** Native (77,247) • Synthetic (76,506) • Translated (416,401)
* **Tasks:** MCQ (single-answer), MCQU (multi-answer), OEQ (open-ended)

> This repository hosts documentation, scripts, and links. Data and models are published on the Hugging Face Hub.

---

## 🚀 Links

### Datasets (Hugging Face)

* **Native:** [https://huggingface.co/datasets/MedInjection-FR/Native](https://huggingface.co/datasets/MedInjection-FR/Native)
* **Synthetic:** [https://huggingface.co/datasets/MedInjection-FR/Synthetic](https://huggingface.co/datasets/MedInjection-FR/Synthetic)
* **Translated:** [https://huggingface.co/datasets/MedInjection-FR/Translated](https://huggingface.co/datasets/MedInjection-FR/Translated)

### Models (Hugging Face)

* **QWEN-4B-NAT:** [https://huggingface.co/MedInjection-FR/QWEN-4B-NAT](https://huggingface.co/MedInjection-FR/QWEN-4B-NAT)
* **QWEN-4B-TRAD:** [https://huggingface.co/MedInjection-FR/QWEN-4B-TRAD](https://huggingface.co/MedInjection-FR/QWEN-4B-TRAD)
* **QWEN-4B-SYN:** [https://huggingface.co/MedInjection-FR/QWEN-4B-SYN](https://huggingface.co/MedInjection-FR/QWEN-4B-SYN)
* **QWEN-4B-NAT-TRAD:** [https://huggingface.co/MedInjection-FR/QWEN-4B-NAT-TRAD](https://huggingface.co/MedInjection-FR/QWEN-4B-NAT-TRAD)
* **QWEN-4B-NAT-SYN:** [https://huggingface.co/MedInjection-FR/QWEN-4B-NAT-SYN](https://huggingface.co/MedInjection-FR/QWEN-4B-NAT-SYN)
* **QWEN-4B-TRAD-SYN:** [https://huggingface.co/MedInjection-FR/QWEN-4B-TRAD-SYN](https://huggingface.co/MedInjection-FR/QWEN-4B-TRAD-SYN)
* **QWEN-4B-ALL (COMBO):** [https://huggingface.co/MedInjection-FR/QWEN-4B-ALL](https://huggingface.co/MedInjection-FR/QWEN-4B-ALL)

---

## 📦 Composition & Splits

| Component  |   Train |    Val |   Test |       Total |
| ---------- | ------: | -----: | -----: | ----------: |
| Native     |  57,563 |  5,055 | 14,629 |      77,247 |
| Synthetic  |  76,506 |      – |      – |      76,506 |
| Translated | 366,370 | 38,011 | 12,020 |     416,401 |
| **Total**  | 500,439 | 43,066 | 26,649 | **570,154** |

Task mix (all components): **OEQ 63,267**, **MCQ 59,597**, **MCQU 454,713**.

---

## 🔧 Quickstart

### Load a dataset (🤗 Datasets)

```python
from datasets import load_dataset

# one of: "Native", "Synthetic", "Translated"
ds = load_dataset("MedInjection-FR/Native")
print(ds)
```

### Run a released model (🤗 Transformers)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "MedInjection-FR/QWEN-4B-NAT-TRAD"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

prompt = (
    "Question: Quelle est la prise en charge initiale d'un OAP ?\n"
    "Choix: A) ... B) ... C) ... D) ...\n"
    "Répondez par la lettre."
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=8)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## 🧪 What’s evaluated

* **MCQ / MCQU:** Exact-Match (EM); MCQU additionally uses **Hamming score**.
  We report both **greedy** and **constrained** decoding (restricted to choice labels).
* **OEQ:** BLEU, ROUGE, METEOR, BERTScore **plus** an **LLM-as-a-judge** calibrated on a **human-annotated** sample (n=100).
  The LLM judge with the best correlation to the physician’s ratings was used for OEQ scoring.

> Mixed training (especially **NAT-TRAD**) showed complementary gains over single-source setups; native data tend to provide the strongest single-source improvements.

---

## 📊 Translated subset quality (WMT’24 biomedical parallel)

| Model             |  BLEU |  COMET |
| ----------------- | ----: | :----: |
| GPT-4o-mini       | 51.01 | 0.8751 |
| Gemini 2.0 Flash  | 53.72 | 0.8783 |
| WMT’24 best (ref) | 53.54 | 0.8760 |

These scores indicate strong translation fidelity for the translated component.

---

## ⚖️ Intended use, ethics, and license

* **Research use only.** Not a substitute for professional medical advice, diagnosis, or treatment.
* No PHI; sources compiled from public educational/exam materials and open datasets.
* Although human checks were performed on a small sample, outputs may still contain errors or biases.
* See the [LICENSE](./LICENSE) for terms. Contact us if your use case is unclear.

---

## 📝 Citation

If you use MedInjection-FR or the released models, please cite:

```bibtex
@inproceedings{medinjection-fr-2025,
  title     = {MedInjection-FR: Investigating Data Provenance for French Biomedical Instruction Tuning},
  author    = {Your Name and Coauthors},
  booktitle = {Proceedings of ...},
  year      = {2025},
  note      = {Datasets and models on the Hugging Face Hub}
}
```

---

## 🤝 Contact

Questions or feedback?

* Open an issue in this GitHub repo
* Or email: [you@example.com](mailto:you@example.com)

---

## 🗺️ Repository structure (suggested)

```
.
├── data_cards/           # component-specific datasheets
├── scripts/              # preprocessing / evaluation helpers
├── configs/              # training configs (DoRA/LoRA etc.)
├── results/              # tables, plots, logs
├── LICENSE
└── README.md
```

---

If you share the exact HF org/user handle and final model IDs, I can tailor the links and the BibTeX entry precisely.

