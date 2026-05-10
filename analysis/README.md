## Mechanistic Analysis of KE

These scripts contain the primary experiments used in Section 5 of the manuscript for identifying Observations 1 to 4. Observations 5 and 6 are qualitative findings, we do not include them in this repository.

---

### Scripts:

#### `obs1.py`

```bash
python obs1.py \
    --model_name <base_model_name> \
    --dataset_name <dataset_name>
```

#### `obs2.py`

```bash
python obs2.py \
    --base_model_name <base_model_name> \
    --edited_model_name <edited_model_name> \
    --dataset_name <dataset_name>
```

#### `obs3.py`

```bash
python obs3.py \
    --base_model_name <base_model_name> \
    --edited_model_name <edited_model_name> \
    --dataset_name <dataset_name>
```

#### `obs4.py`

```bash
python obs4.py \
    --base_model_name <base_model_name> \
    --edited_model_name <edited_model_name> \
    --dataset_name <dataset_name> \
    --implicit_dataset_name <implicit_dataset_name>
```

---

### Dataset Format

#### Main Dataset

```json
{
    "paraphrases": ["prompt_1", "prompt_2"],
    "old_edit": "original fact",
    "new_edit": "edited fact",
    "subject": "entity"
}
```

#### Implicit Reasoning Dataset

```json
{
    "prompt": "reasoning question",
    "old_reason": "original reasoning answer",
    "new_reason": "edited reasoning answer",
    "subject": "entity"
}
```
