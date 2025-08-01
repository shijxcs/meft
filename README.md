# Memory-Efficient Fine-Tuning via Low-Rank Activation Compression

## Usage

Simply replace the original Trainer with `MeftTrainer` and add the configuration:

```python
from meft import MeftConfig, MeftTrainer

trainer = MeftTrainer(
    ...,
    meft_config=MeftConfig(...),
)
```

For trainer variants (e.g. SFTTrainer), use the subscript syntax:
```python
from datasets import load_dataset
from meft import MeftConfig, MeftTrainer
from trl import SFTTrainer

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train[:1%]")

trainer = MeftTrainer[SFTTrainer](
    model="Qwen/Qwen3-0.6B-Base",
    train_dataset=dataset,
    meft_config=MeftConfig(
        patch_locations="layer",
        compress_kwargs={"rank": 128},
    ),
)
trainer.train()
```

Please refer to [config.py](config.py) for detailed configurations.
