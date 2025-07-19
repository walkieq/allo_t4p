import torch
import torch.nn as nn
import torch.nn.functional as F
import allo
import os
import numpy as np
from time import time

cur_dir = os.path.dirname(os.path.abspath(__file__))
allo_t4p_dir = os.path.dirname(os.path.dirname(cur_dir))
llvm_build_dir = os.path.join(allo_t4p_dir, "externals/llvm-project/build")
os.environ["LLVM_BUILD_DIR"] = llvm_build_dir


class Particle_MLP(nn.Module):
    def __init__(self):
        super(Particle_MLP, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.output(x), dim=-1)
        return x


batch_size = 32
# num_particles = 8
num_feats = 16

model = Particle_MLP().eval()
example_inputs = [torch.rand(batch_size, num_feats)]

# generate HLS model
t = time()
mode = "sw_emu"
os.environ["XDEVICE"] = mode
vitis_mod = allo.frontend.from_pytorch(
    model,
    example_inputs=example_inputs,
    target="vitis_hls",
    mode=mode,
    project="mlp.prj",
)
print(vitis_mod.hls_code)
print(f"Time taken for {mode}: {time() - t}s")


golden = model(*example_inputs)
x_np = np.random.random((batch_size, num_feats)).astype(np.float32)
allo_out = np.zeros((batch_size, 5), dtype=np.float32)

vitis_mod(x_np, allo_out)
np.testing.assert_allclose(allo_out, x_np, rtol=1e-5, atol=1e-5)
print("Passed!")
