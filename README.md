# Akkadian → English Machine Translation  
**Deep Past Challenge (Kaggle)**

<p align="center">
  <img src="plots/plot6_akkadian_wordcloud.png" width="420" alt="Akkadian word cloud">
  <img src="plots/plot7_english_wordcloud.png" width="420" alt="English word cloud">
</p>

This project investigates neural machine translation of Old Assyrian cuneiform texts from Akkadian transliteration into English. The corpus consists primarily of commercial and administrative documents from ancient Mesopotamian trade networks, dating to approximately 1900–1700 BCE.

The goal is to evaluate how modern sequence-to-sequence transformer models perform on carefully preprocessed and sentence-aligned Akkadian–English parallel data.

---

## 1. Data

### 1.1 Sources

<p align="left">
  <img src="plots/plot1_data_sources_pie.png" width="260" alt="Data sources pie chart">
</p>

Training data was assembled from two sources:

1. **Akkademia** — a pre-aligned Akkadian–English parallel corpus  
2. **Deep Past Kaggle competition data** — official training material

After preprocessing and alignment, the final dataset contains **43,746 sentence-level pairs**.

---

## 2. Preprocessing

Akkadian transliterations include numerous editorial conventions and scribal annotations that are not suitable for direct input to neural machine translation models. A custom preprocessing pipeline was implemented to normalize the text.

### 2.1 Scribal notation removal

The following elements were removed:
- line numbers (`1`, `5`, `10`, `1'`, …)  
- certainty markers (`!`, `?`)  
- insertions (`<text>`)  
- damaged sign markers (`˹˺`)  
- gap indicators (`[x]`, `...`, `<gap>`)

### 2.2 Determinatives

Determinatives such as `{d}` (deity), `{ki}` (place), and `{m}` (person) function as semantic classifiers rather than lexical items. These were removed entirely to reduce sparsity and noise in the training data.

### 2.3 Character normalization

Orthographic variation and special characters were normalized to consistent ASCII-friendly forms:
- `á → a2`, `š → sz`, `ṣ → s,`, etc.

**Example**
```text
Before: 5 a-na {m}A-šùr-ma-lik qí-bí-ma [!]
After:  a-na A-szur-ma-lik qi2-bi-ma
````

---

## 3. Sentence-level alignment

The evaluation format of the competition is sentence-based, whereas part of the training data is document-level. Automatic sentence splitting and alignment was performed using:

* line structure in Akkadian transliterations
* punctuation-based splitting for English
* retention only of cases with clean 1-to-1 correspondence

This process increased the effective training data from approximately 1,500 documents to **43,746 aligned sentence pairs**.

---

## 4. Data characteristics

### 4.1 Length distributions

![Word Length Distribution](plots/plot3_word_length_histogram.png)

English translations are typically longer than Akkadian transliterations, with a median length ratio of **1.58:1**.

![Translation Length Ratio](plots/plot5_length_ratio.png)

**Summary statistics**

* Avg. Akkadian words: **13.1** per sentence
* Avg. English words: **21.7** per sentence
* Avg. Akkadian characters: **105.9**
* Avg. English characters: **120.4**

### 4.2 Vocabulary

![Vocabulary Comparison](plots/plot8_vocabulary_comparison.png)

* Akkadian vocabulary size: **74,114** unique tokens
* English vocabulary size: **46,588** unique tokens
* Vocabulary diversity (Akkadian): **0.129**
* Vocabulary diversity (English): **0.049**

The higher diversity in Akkadian reflects morphological richness and transliteration variability.

### 4.3 Frequent Akkadian tokens

![Top Akkadian Words](plots/plot9_top_akkadian_words.png)

High-frequency terms are dominated by prepositions, function words, and administrative vocabulary, consistent with the commercial nature of the corpus.

---

## 5. Model

The translation model is based on **NLLB-200 (distilled 600M)**
(`facebook/nllb-200-distilled-600M`).

Initial experiments with a smaller model showed early performance saturation. The 600M-parameter NLLB model provided sufficient capacity and strong multilingual representations to adapt to the domain-specific characteristics of Akkadian transliteration.

---

## 6. Training

**Configuration**

* batch size: **6** (per device)
* gradient accumulation: **4** (effective batch size 24)
* learning rate: **3e-5** with warmup
* optimizer: **AdamW**
* epochs: **3**
* precision: **FP16**

**Evaluation metrics**

* competition metric: geometric mean of **BLEU** and **chrF++**
* BLEU and chrF++ tracked individually

---

## 7. Results

**Test set**

* Geometric mean: **[TO BE FILLED]**
* BLEU: **[TO BE FILLED]**
* chrF++: **[TO BE FILLED]**
* Leaderboard position: **[TO BE FILLED]**

| Model            | Geometric Mean | BLEU  | chrF++ |
| ---------------- | -------------- | ----- | ------ |
| Smaller baseline | [TBF]          | [TBF] | [TBF]  |
| NLLB-200 (600M)  | [TBF]          | [TBF] | [TBF]  |

---

## 8. Examples

```text
Akkadian: KISZIB ma-nu-ba-lu2m-a-szur DUMU s,i2-la2-(d)IM ...
English:  Seal of Mannum-balum-Aššur son of Ṣilli-Adad, ...
```

```text
Akkadian: TÚG sza qa2-tim i-tur4-DINGIR il5-qe2...
English:  Itūr-ilī has received one textile of ordinary quality....
```

```text
Akkadian: TÚG u-la i-di2-na-ku-um i-tu3-ra-ma 9 GÍN KÙ.BABBAR...
English:  he did not give you a textile. He returned and 9 shekels of silver...
```

---

## 9. Reproducibility

```bash
python train.py \
  --model facebook/nllb-200-distilled-600M \
  --data final_training_data.csv \
  --batch_size 6 \
  --grad_accum 4 \
  --lr 3e-5 \
  --epochs 3
```

---

## 10. References

1. NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation.* arXiv:2207.04672
2. Akkademia (Gutherz et al.), GitHub repository
3. Deep Past Initiative, Kaggle competition

---

## Dataset summary

![Summary Statistics](plots/plot10_summary_table.png)

