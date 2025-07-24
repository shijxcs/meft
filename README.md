# MEFT

Memory-Efficient Fine-Tuning via Low-Rank Activation Compression

## Usage

Simply replace the original Trainer with `MeftTrainer` and add the configuration:

```python
from transformers import Trainer
from meft import MeftConfig, MeftTrainer

trainer = MeftTrainer(
    ...,
    meft_config=MeftConfig(
        patch_locations="sublayer",
        compress_kwargs={"rank": 128},
    ),
)
```

For trainer variants (e.g. SFTTrainer), use the subscript syntax:
```python
from trl import SFTTrainer
from meft import MeftConfig, MeftTrainer

trainer = MeftTrainer[SFTTrainer](
    ...
    meft_config=MeftConfig(
        patch_locations="sublayer",
        compress_kwargs={"rank": 128},
    ),
)
```

Please refer to [config.py](config.py) for configurations details.
