# Akkadian → English Machine Translation  (Work in progress/progress log)
**Deep Past Challenge (Kaggle)**

<table width="100%" cellspacing="0" cellpadding="0">
  <tr>
    <td width="50%" style="padding:0; margin:0;">
      <img src="plots/plot6_akkadian_wordcloud.png" width="100%" alt="Akkadian word cloud">
    </td>
    <td width="50%" style="padding:0; margin:0;">
      <img src="plots/plot7_english_wordcloud.png" width="100%" alt="English word cloud">
    </td>
  </tr>
</table>

This project investigates neural machine translation of Old Assyrian cuneiform texts from Akkadian transliteration into English.
The goal is to evaluate how modern sequence-to-sequence transformer models perform on carefully preprocessed and sentence-aligned Akkadian–English parallel data.
Current status: I’m currently 182nd on the public leaderboard (33.4). At this point, further progress seems to depend heavily on time-consuming data cleaning and alignment work,
which I don’t have the time to prioritize right now, so it’s unclear if I’ll keep pushing this further.

This project gave practical experience adapting translation models to a low-resource, 
transliteration-heavy domain. Working with byte-level models (ByT5) and different fine-tuning regimes made it clear how strongly preprocessing, 
domain shift, and decoding choices influence BLEU/chrF performance. It also reinforced that some of the hardest bottlenecks are data- and convention-driven, and are difficult to fully resolve without domain expertise.

# Code
byt5_two_stage_training: Trains a ByT5 model in two phases—first on a broader Akkadian dataset, then fine-tunes on the competition (Old Assyrian).
submission_notebook: Loads a selected checkpoint and runs inference on the test set using tuned decoding parameters, then formats and writes the Kaggle submission file.

---
##  Training Log
 - NLLB-200 + Akkademia (out-of-period extra data): Adding a large external Akkademia-derived dataset (from a different historical period than the Kaggle corpus) leaderboard performance of 23.2, 
 the extra data helped the model’s general translation robustness despite the domain mismatch. This differs from what others have reported.
- ByT5-Small + Akkademia + extracted extra data: Combining a smaller byte-level T5 model with Akkademia and the host-provided extracted pairs gave solid local gains, a leaderboard score of around 30.
- ByT5-Small fine-tuned on competition training data (Old Assyrian only): Starting from the general Akkadian checkpoint and continuing training only on the competition (Old Assyrian) data with a lower learning rate produced a clear leaderboard jump.
---

## 1. Data

### 1.1 Sources
- Deep Past Kaggle competition data official training material
- Akkademia: a pre-aligned Akkadian–English parallel corpus. Not old Akkadian.
- Extracted pairs: additional Akkadian transliteration–translation snippets automatically extracted from digitized sources, provided by the competition hosts and used as supplemental training data (quality varies).

---
### 1.2 Data quality
During development, it became clear that the training data contains several sources of noise and inconsistency that are difficult to fully resolve from a machine-learning perspective alone. 
Community discussions in the competition indicate that a non-trivial fraction of the training examples may be partially misaligned or truncated.
While filtering or cropping can reduce some of these effects, identifying and correcting such issues reliably often requires domain-specific knowledge.

More broadly, the dataset reflects multiple scholarly transliteration conventions and editorial practices, including variation in diacritics, orthography, and annotation. 
Resolving these ambiguities robustly is, agin, primarily a domain-expert task.

## 2. Preprocessing

Akkadian transliterations include numerous editorial conventions and scribal annotations that are not suitable for direct input to neural machine translation models. A custom preprocessing pipeline was implemented to normalize the text.

### 2.1 Scribal notation removal

## 2. Preprocessing

Akkadian transliterations include editorial conventions (line numbers, damage markers, brackets, etc.) that are not suitable for direct input to neural machine translation models. I implemented a lightweight preprocessing class to normalize the text before sentence splitting and training.

**Note on implementation details.** Parts of the character-mapping table (e.g., Unicode subscripts and diacritics to normalized ASCII-friendly forms) were generated with the help of an LLM from written specifications, mainly to avoid manually typing a long mapping. If you reuse this code for other corpora or transliteration conventions, it is worth double-checking the mappings against your expected standard.

Determinatives such as `{d}` (deity), `{ki}` (place), and `{m}` (person) function as semantic classifiers rather than lexical items. These were removed entirely to reduce sparsity and noise in the training data.
Orthographic variation and special characters were normalized to consistent ASCII-friendly forms:
- `á → a2`, `š → sz`, `ṣ → s,`, etc.

**Example**
```text
Before: 5 a-na {m}A-šùr-ma-lik qí-bí-ma [!]
After:  a-na A-szur-ma-lik qi2-bi-ma
````
When traning NLLB-200, I used the following preprocessing
```python
class AkkadianPreprocessor:
    """Clean and preprocess Akkadian texts.
       Optional: lexicon-based PN/GN tagging."""
    
    def __init__(self, lexicon_df=None):
        self.lexicon = lexicon_df
        self.has_lexicon = lexicon_df is not None

        self.char_replacements = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9', '₊': 'x',
            'á': 'a2', 'à': 'a3', 'é': 'e2', 'è': 'e3',
            'í': 'i2', 'ì': 'i3', 'ú': 'u2', 'ù': 'u3',
            'š': 'sz', 'Š': 'SZ', 'ṣ': 's,', 'Ṣ': 'S,',
            'ṭ': 't,', 'Ṭ': 'T,', 'ḫ': 'h', 'Ḫ': 'H', 'ʾ': "'",
        }

        if self.has_lexicon:
            self._build_lexicon_lookups()

    def _build_lexicon_lookups(self):
        self.form_to_type = {}
        for _, row in self.lexicon.iterrows():
            form = str(row.get('form', '')).strip().lower()
            typ = row.get('type', None)
            if form and form != 'nan' and pd.notna(typ):
                self.form_to_type[form] = str(typ).strip()

        self.person_names = {f for f, t in self.form_to_type.items() if t == 'PN'}
        self.place_names  = {f for f, t in self.form_to_type.items() if t == 'GN'}

    def clean_text(self, text, is_akkadian=True, tag_proper_nouns=False):
        if pd.isna(text) or str(text).strip() == "":
            return ""

        text = str(text)

        # remove line numbers
        text = re.sub(r'(?:^|\n)\s*\d+\'*\s*', ' ', text)

        # remove scribal notations
        text = text.replace('!', '').replace('?', '').replace('/', ' ')
        text = re.sub(r'<([^>]+)>', r'\1', text)
        text = re.sub(r'<<([^>]+)>>', r'\1', text)
        text = re.sub(r'[˹˺]', '', text)
        text = re.sub(r'\[([^\]]+)\]', r'\1', text)

        # remove gaps / ellipses entirely
        text = re.sub(r'<big[_\s]?gap>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<gap>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[x\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\.\.\.', '', text)
        text = text.replace('…', '')

        # remove determinatives completely
        text = re.sub(r'\{[^}]+\}', '', text)

        # normalize characters (akkadian only)
        if is_akkadian:
            for old, new in self.char_replacements.items():
                text = text.replace(old, new)

        # whitespace normalize early so tokenization is stable
        text = re.sub(r'\s+', ' ', text).strip()

        # optional: tag PN/GN using lexicon
        if tag_proper_nouns and self.has_lexicon and is_akkadian:
            words = text.split()
            out = []
            for w in words:
                key = w.replace('-', '').lower()
                if key in self.person_names:
                    out.append(f"[PN]{w}[/PN]")
                elif key in self.place_names:
                    out.append(f"[GN]{w}[/GN]")
                else:
                    out.append(w)
            text = ' '.join(out)

        return text
```
For traning ByT5 base simpler preprocessing was used:
```python
ASCII_TO_DIACRITIC = {
    "sz": "š", "SZ": "Š", "Sz": "Š", "sh": "š", "SH": "Š", "Sh": "Š",
    "s,": "ṣ", "S,": "Ṣ", "t,": "ṭ", "T,": "Ṭ", "z,": "ẓ", "Z,": "Ẓ",
    ".s": "ṣ", ".S": "Ṣ", ".t": "ṭ", ".T": "Ṭ", ".z": "ẓ", ".Z": "Ẓ",
    "h,": "ḫ", "H,": "Ḫ", ".h": "ḫ", ".H": "Ḫ", "hh": "ḫ", "HH": "Ḫ",
    "s2": "š", "S2": "Š", "s3": "ś", "S3": "Ś",
    "a2": "á", "a3": "à", "e2": "é", "e3": "è",
    "i2": "í", "i3": "ì", "u2": "ú", "u3": "ù",
}

SUBSCRIPTS = {'₀':'0', '₁':'1', '₂':'2', '₃':'3', '₄':'4',
              '₅':'5', '₆':'6', '₇':'7', '₈':'8', '₉':'9', 'ₓ':'x'}

def normalize_ascii(text):
    if not text:
        return text
    for k, v in sorted(ASCII_TO_DIACRITIC.items(), key=lambda x: -len(x[0])):
        text = text.replace(k, v)
    for k, v in SUBSCRIPTS.items():
        text = text.replace(k, v)
    return text

def normalize_gaps(text):
    if not text:
        return text
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i].lower() == "x":
            count = 1
            while i + count < len(tokens) and tokens[i + count].lower() == "x":
                count += 1
            result.append("<gap>" if count == 1 else "<big_gap>")
            i += count
        else:
            t = tokens[i]
            if t.lower().startswith("x-"):
                t = "<gap>" + t[1:]
            elif t.lower().endswith("-x"):
                t = t[:-1] + "-<gap>"
            result.append(t)
            i += 1
    text = " ".join(result)
    text = re.sub(r"(<gap>\s*){2,}", "<big_gap> ", text)
    text = re.sub(r"\.\.\.+", " <big_gap> ", text)
    return text.strip()

def tag_names(text):
    if not USE_LEXICON or not text:
        return text
    words = text.split()
    result = []
    for w in words:
        key = w.replace("-", "").lower()
        if key in PN_NAMES:
            result.append(f"[PN]{w}[/PN]")
        elif key in GN_NAMES:
            result.append(f"[GN]{w}[/GN]")
        else:
            result.append(w)
    return " ".join(result)

def clean_akkadian(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text)
    text = text.replace("!", "").replace("?", "")
    text = re.sub(r"[˹˺]", "", text)
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    text = normalize_ascii(text)
    text = normalize_gaps(text)
    text = tag_names(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```


---

## 4. Data characteristics

### 4.1 Length distributions

**Summary statistics**

* Avg. Akkadian words: **13.1** per sentence
* Avg. English words: **21.7** per sentence
* Avg. Akkadian characters: **105.9**
* Avg. English characters: **120.4**

### 4.2 Vocabulary

* Akkadian vocabulary size: **74,114** unique tokens
* English vocabulary size: **46,588** unique tokens
* Vocabulary diversity (Akkadian): **0.129**
* Vocabulary diversity (English): **0.049**

The higher diversity in Akkadian reflects morphological richness and transliteration variability.

---


# Data Augmentation: Expanding the Akkadian Training Corpus (work in progress)

## Overview

The original Kaggle dataset contained approximately 1,500 Akkadian transliteration-translation pairs. To improve model performance, I undertook a substantial data augmentation effort to extract additional training pairs from source material
provided by the organisers. The data was provided as a large set of OCR-extracted text from academic PDFs.  The data extraction process proved challenging and required multiple iterations of extraction, cleaning, and manual review. T

## Initial Attempts: Rule-Based Extraction

My first approach was to use traditional rule-based methods and standard NLP techniques directly on OCR-extracted text from academic PDFs. This involved:

- Pattern matching for Akkadian transliteration markers (hyphenated syllables, Sumerograms, diacritics like š, ṣ, ṭ, ḫ)
- Heuristics to identify translation sections
- Language detection to separate transliteration from translation

**Result:** This approach did not work well. Simple pattern matching couldn't reliably separate the components.

## LLM-Based Extraction

Given the limitations of rule-based approaches, I wanted to test using a Large Language Model for extraction. 
I set up a cloud-based virtual machine to run the extraction at scale.

### Model and Configuration

- **Model:** Qwen/Qwen2.5-7B-Instruct

### Extraction Prompt

```
SYSTEM PROMPT:
You are an expert Assyriologist extracting data from scholarly publications.

Your task: Find Akkadian TRANSLITERATIONS and their TRANSLATIONS on the page.

DEFINITIONS:
- TRANSLITERATION: Akkadian text written in Latin letters with hyphens, diacritics, line numbers.
  Example: "1. um-ma A-šur-i-dí-ma 2. a-na Pù-šu-ki-in qí-bi-ma"
- TRANSLATION: A modern language rendering of the Akkadian text (English, German, French).
  Example: "Thus says Ashur-idi to Pushukin: ..."

NOT translations: Scholarly discussion, historical commentary, footnotes.

OUTPUT FORMAT - Return ONLY this JSON:
{"status": "...", "confidence": 0.X, "transliteration_raw": "...", "translation_raw": "..."}

STATUS must be one of:
- "good_pair": Found BOTH transliteration AND its translation
- "transliteration_only": Found transliteration but NO translation on this page
- "translation_only": Found translation but NO transliteration on this page  
- "junk": No transliteration or translation found (bibliography, index, prose)

CRITICAL RULES:
1. Copy text VERBATIM - do not modify, clean, or translate
2. Include line numbers if present (e.g., "1. um-ma... 2. a-na...")
3. If status is "transliteration_only", translation_raw MUST be ""
4. If status is "translation_only", transliteration_raw MUST be ""
5. If status is "junk", BOTH fields MUST be ""
6. Scholarly discussion is NOT a translation - mark as "junk" unless actual transliteration present
```

### Extraction Results

Processing approximately 217,000 PDF pages over ~31 hours yielded:

| Category | Count | Percentage |
|----------|-------|------------|
| Pages processed | ~217,000 | 100% |
| Candidate pages (with Akkadian indicators) | ~71,000 | 33% |
| Good pairs extracted | ~7,100 | 10% of candidates |
| Transliteration only | ~12,200 | 17% of candidates |
| Junk/rejected | ~49,300 | 70% of candidates |

The high junk rate was expected—most pages in academic publications contain bibliography, indices, commentary, or other non-transliteration content.

## Post-Processing Pipeline

The raw extraction output required substantial cleaning. Many extracted pairs had issues:

- **Mixed content:** Translation text embedded within the transliteration field (common in PDFs where translation follows immediately after transliteration)
- **Multiple languages:** Translations in French, German, Turkish, Italian (not just English)
- **OCR artifacts:** Corrupted characters and formatting issues
- **Duplicates:** The extraction process created multiple copies of some records

### LLM-Based Cleaning Attempt

I first attempted to use the same LLM to clean problematic records—specifically those where translation content appeared mixed into the transliteration field. 
The prompt asked the model to separate the components.

**Result:** This approach performed poorly. The LLM tended to produce interleaved output (word-by-word glosses) rather than cleanly separated fields. 
Of approximately 2,700 LLM-cleaned records, only 7 passed subsequent validation checks.

### Rule-Based Cleaning System

I developed a rule-based cleaning pipeline with the following components:

1. **Language detection:** Identifying the language of translations using character patterns and word lists for Turkish, German, French, Italian, Spanish, Dutch, and English
2. **Mixed content detection:** Identifying when translation text appeared in the transliteration field by detecting language-specific markers
3. **Content splitting:** Attempting to find boundaries between transliteration and translation sections
4. **Translation:** Converting non-English translations to English using Helsinki-NLP's MarianMT models
5. **Validation:** Checking for remaining contamination and quality issues

After rule-based processing:
- **Total pairs:** 5,772
- **Needing manual review:** 3,011

## Manual Review Interface

To handle the records flagged for review, I built a web interface using Flask that displayed transliteration and translation side-by-side for manual approval and cleaning. 
This allowed me to:
- Approve clean pairs
- Edit and correct minor issues
- Reject pairs with unfixable problems
- Identify patterns in problematic extractions

I manually reviewed all 3,011 flagged pairs through this interface.

## Final Cleaning

Even after manual review, some issues remained. A final automated check identified records with residual language contamination:

**Source (transliteration) contamination:**
| Language | Count |
|----------|-------|
| Italian | 94 |
| French | 77 |
| German | 43 |
| Turkish | 17 |

**Target (translation) contamination:**
| Language | Count |
|----------|-------|
| French | 6 |
| Italian | 3 |
| German | 1 |
| Turkish | 1 |

These contaminated records were removed from the final dataset.

## Final Results

| Dataset | Pairs |
|---------|-------|
| Original Kaggle data | ~1,500 |
| New extracted pairs | 4,405 |
| **Total training data** | **~5,900** |

The extraction pipeline efficiency was admittedly poor—starting from ~7,100 extracted pairs and ending with 4,405 clean pairs represents significant loss. Several factors contributed:

1. **Aggressive filtering:** I opted to be strict rather than risk including junk data that could harm model training
2. **Mixed content challenges:** Academic PDFs frequently place translations immediately after transliterations with no clear separator
3. **Multilingual complexity:** The corpus includes scholarship in multiple languages, complicating both extraction and cleaning
4. **OCR quality:** Variable scan quality across different publications

Despite these challenges, the augmentation effort more than doubled the available training data. However, at this stage, it is not clear that the
quality of the data is good enough to improve traning.

Future improvements could include:
- Better extraction prompts with few-shot examples
- Improved boundary detection for mixed content
- Earlier deduplication in the pipeline
- More sophisticated language-aware splitting

  
---


## 5. Model

 -NLLB-200 (distilled 600M) (facebook/nllb-200-distilled-600M) is a multilingual seq2seq translation model pretrained on many language pairs, designed to transfer strong cross-lingual representations to low-resource translation. It uses subword tokenization and is optimized for standard machine translation settings.
- ByT5 is a T5-style seq2seq model that operates at the byte/character level instead of subwords. This makes it more robust to unusual symbols, diacritics, and inconsistent formatting common in transliteration-heavy inputs like Akkadian.


1. S. a-na a-wa-tim a-ni-a-tim !up-pu-ti 10 i-li-ku-ni-im a-wa-ti / lâ tzi-u.-ta-ki-ı /
2. Ennum-Assur, an Nuhsatum: Warum schreibst du mir immer wieder dumme Worte?

