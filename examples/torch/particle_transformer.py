import allo
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import ConstituentNet

num_particles = 8
num_feats = 3
num_transformers = 1
embbed_dim = 64
num_heads = 2
dropout = 0
batch_size = 16

model_config = {
    "in_dim": num_feats,
    "embbed_dim": embbed_dim,
    "num_heads": num_heads,
    "num_classes": 5,
    "num_transformers": num_transformers,
    "dropout": dropout,
    "num_particles": num_particles,
    "activation": "ReLU",
    "normalization": "Batch",
}

module = ConstituentNet(**model_config).eval()
example_inputs = [torch.rand(batch_size, num_particles, num_feats)]
golden = module(*example_inputs)
llvm_mod = allo.frontend.from_pytorch(
    module,
    example_inputs=example_inputs,
    verbose=True,
)

golden = module(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
np.testing.assert_allclose(res, golden.detach().numpy(), atol=1e-3)
print("Test passed!")

# generate HLS module
# mod = allo.frontend.from_pytorch(module, example_inputs=example_inputs, target="vhls")
# print(mod.hls_code)
