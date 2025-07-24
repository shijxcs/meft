import torch
import torch._dynamo

torch.set_float32_matmul_precision('high')
# torch._dynamo.config.disable = True
# torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 256

from .trainer import MeftConfig, MeftTrainer
