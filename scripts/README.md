# Annotation scripts

Trankit-based annotation pipeline for three Slovenian corpora: **ccKres**, **GOS** and **Solar**.
All scripts are run from the repository root. The same model is used for all corpora:
`save_dir_ssj2.14+sst2.15-stan+pog` with `xlm-roberta-large`.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `annotate.py` | Generic Trankit annotator. Reads pre-tokenised CoNLL-U, runs Trankit in chunks, writes predictions. Shared by all three corpora. |
| `annotate_all.sh` | Runs annotation for all three corpora sequentially. Edit `CHUNK_SIZE` at the top to tune throughput. |
| `merge_space_after.py` | Copies `SpaceAfter=No` from an original CoNLL-U into a Trankit-annotated CoNLL-U. Writes to a new file — never overwrites input. |
| `merge_space_after_all.sh` | Runs `merge_space_after.py` for all three corpora, writing `*_annotated_spaceafter.conllu` files. |
| `cckres/cckres_tei_to_conllu.py` | Converts ccKres TEI XML (`cckresV1_0/*.xml`) to CoNLL-U with original tokenisation and `SpaceAfter=No` from `<S/>` elements. |
| `cckres/cckres_annotate.sh` | Annotates ccKres only (wraps `annotate.py`). |
| `gos/gos_tei_to_conllu.py` | Converts GOS TEI XML (`Gos.TEI/**/*.xml`) to CoNLL-U with original tokenisation and `SpaceAfter=No` from `join="right"` attributes. File list is read from `Gos2.1.xml`. |
| `gos/gos_annotate.sh` | Annotates GOS only (wraps `annotate.py`). |
| `solar/solar_annotate.sh` | Annotates Solar only (wraps `annotate.py`). Solar is already in CoNLL-U — no conversion needed. |

---

## Required data

| Corpus | Required input | Location |
|--------|---------------|----------|
| ccKres | Extracted TEI XML files | `data/datasets/ccKres/cckresV1_0/*.xml` |
| GOS | Extracted TEI XML files + manifest | `data/datasets/GOS/Gos.TEI/` |
| Solar | Original CoNLL-U | `data/datasets/solar/Solar.CoNLL-U/solar-orig.conllu` |
| All | Trained model | `data/models/save_dir_ssj2.14+sst2.15-stan+pog/` |

---

## Execution order

### ccKres
```bash
# Step 1 — convert TEI to CoNLL-U
source venv/bin/activate
python scripts/cckres/cckres_tei_to_conllu.py

# Step 2 — annotate with Trankit (or use annotate_all.sh for all corpora)
bash scripts/cckres/cckres_annotate.sh

# Step 3 — merge SpaceAfter back in
python scripts/merge_space_after.py \
    --original  data/datasets/ccKres/cckres.conllu \
    --annotated data/datasets/ccKres/cckres_annotated.conllu \
    --output    data/datasets/ccKres/cckres_annotated_spaceafter.conllu
```

### GOS
```bash
# Step 1 — convert TEI to CoNLL-U
python scripts/gos/gos_tei_to_conllu.py

# Step 2 — annotate
bash scripts/gos/gos_annotate.sh

# Step 3 — merge SpaceAfter
python scripts/merge_space_after.py \
    --original  data/datasets/GOS/gos.conllu \
    --annotated data/datasets/GOS/gos_annotated.conllu \
    --output    data/datasets/GOS/gos_annotated_spaceafter.conllu
```

### Solar
```bash
# Step 1 — annotate (no conversion needed)
bash scripts/solar/solar_annotate.sh

# Step 2 — merge SpaceAfter
python scripts/merge_space_after.py \
    --original  data/datasets/solar/Solar.CoNLL-U/solar-orig.conllu \
    --annotated data/datasets/solar/solar_annotated.conllu \
    --output    data/datasets/solar/solar_annotated_spaceafter.conllu
```

### All corpora at once
```bash
source venv/bin/activate
bash scripts/annotate_all.sh 2>&1 | tee scripts/annotate_all.out
bash scripts/merge_space_after_all.sh
```
