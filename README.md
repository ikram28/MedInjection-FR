# MedInjection-FR

A French biomedical **instruction dataset** and **model suite** for studying how data provenance (native, synthetic, translated) impacts instruction-tuning of LLMs.

* **Total size:** 570,154 instruction–response pairs
* **Components:** Native (77,247) • Synthetic (76,506) • Translated (416,401)
* **Tasks:** MCQ (single-answer), MCQU (multi-answer), OEQ (open-ended)

> This repository hosts documentation, scripts, and links. Data and models are published on the Hugging Face Hub.

---

## 🚀 Links

### Datasets 🤗

* **Native:** [https://huggingface.co/datasets/MedInjection-FR/Native](https://huggingface.co/datasets/MedInjection-FR/Native)
* **Synthetic:** [https://huggingface.co/datasets/MedInjection-FR/Synthetic](https://huggingface.co/datasets/MedInjection-FR/Synthetic)
* **Translated:** [https://huggingface.co/datasets/MedInjection-FR/Translated](https://huggingface.co/datasets/MedInjection-FR/Translated)

### Models 🤗

* **QWEN-4B-NAT:** [https://huggingface.co/MedInjection-FR/QWEN-4B-NAT](https://huggingface.co/MedInjection-FR/QWEN-4B-NAT)
* **QWEN-4B-TRAD:** [https://huggingface.co/MedInjection-FR/QWEN-4B-TRAD](https://huggingface.co/MedInjection-FR/QWEN-4B-TRAD)
* **QWEN-4B-SYN:** [https://huggingface.co/MedInjection-FR/QWEN-4B-SYN](https://huggingface.co/MedInjection-FR/QWEN-4B-SYN)
* **QWEN-4B-NAT-TRAD:** [https://huggingface.co/MedInjection-FR/QWEN-4B-NAT-TRAD](https://huggingface.co/MedInjection-FR/QWEN-4B-NAT-TRAD)
* **QWEN-4B-NAT-SYN:** [https://huggingface.co/MedInjection-FR/QWEN-4B-NAT-SYN](https://huggingface.co/MedInjection-FR/QWEN-4B-NAT-SYN)
* **QWEN-4B-TRAD-SYN:** [https://huggingface.co/MedInjection-FR/QWEN-4B-TRAD-SYN](https://huggingface.co/MedInjection-FR/QWEN-4B-TRAD-SYN)
* **QWEN-4B-ALL:** [https://huggingface.co/MedInjection-FR/QWEN-4B-ALL](https://huggingface.co/MedInjection-FR/QWEN-4B-ALL)

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
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "MedInjection-FR/QWEN-4B-NAT-TRAD"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = """Un professionnel de santé de 54 ans consulte un spécialiste des maladies infectieuses pour un suivi concernant un diagnostic récent d'hépatite C chronique. 
          Il s'est initialement présenté avec des symptômes tels que fatigue, malaise et enzymes hépatiques élevées et soupçonne d'avoir contracté l'infection à la suite
          d'une piqûre d'aiguille il y a des années. Malgré le début du traitement, son titre viral reste élevé, ce qui incite le médecin à ajouter un nouveau médicament
          qui inhibe la maturation virale en bloquant la synthèse des protéines. Quel est l'effet indésirable le plus probable de ce médicament ?
          Choix de réponses : 
          (A) Uropathie cristalline obstructive 
          (B) Suppression de la moelle osseuse 
          (C) Insomnie et irritabilité 
          (D) ..."""
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)

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

```

---

## 🤝 Contact

Questions or feedback?

* Open an issue in this GitHub repo
* Or email: [you@example.com](mailto:you@example.com)







