import torch
import torch.nn as nn
import torch.nn.functional as F
import allo
import os
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
allo_t4p_dir = os.path.dirname(os.path.dirname(cur_dir))
llvm_build_dir = os.path.join(allo_t4p_dir, "externals/llvm-project/build")
os.environ["LLVM_BUILD_DIR"] = llvm_build_dir

# source ~/xilinx_vitis.sh
# source /opt/xilinx/xrt/setup.sh


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

# Model weights will be initialized randomly each time
# So rebuild the hls project each time to compare the results with the golden output
# Or load the same model weights to compare the results
model = Particle_MLP().eval()
example_inputs = [torch.rand(batch_size, num_feats)]


# LLVM
# llvm_mod = allo.frontend.from_pytorch(
#     model, example_inputs=example_inputs, verbose=False
# )
# golden = model(*example_inputs)
# np_inputs = [x.detach().numpy() for x in example_inputs]
# res = llvm_mod(*np_inputs)
# torch.testing.assert_close(res, golden.detach().numpy(), rtol=1e-5, atol=1e-5)
# print("Passed!")


# VITIS HLS
mode = "sw_emu"
os.environ["XDEVICE"] = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
os.environ["XCL_EMULATION_MODE"] = mode

vitis_mod = allo.frontend.from_pytorch(
    model,
    example_inputs=example_inputs,
    target="vitis_hls",
    mode=mode,
    project="mlp_sw.prj",
)
# print(vitis_mod.hls_code)

golden = model(*example_inputs)
x_np = x_np = example_inputs[0].detach().numpy().astype(np.float32)
allo_out = np.zeros((batch_size, 5), dtype=np.float32)

vitis_mod(x_np, allo_out)
np.testing.assert_allclose(allo_out, golden.detach().numpy(), rtol=1e-5, atol=1e-5)
print("Passed!")
