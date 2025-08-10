import allo
import torch
import numpy as np
import os
from model import ConstituentNet, SliceClsToken

cur_dir = os.path.dirname(os.path.abspath(__file__))
allo_t4p_dir = os.path.dirname(os.path.dirname(cur_dir))
llvm_build_dir = os.path.join(allo_t4p_dir, "externals/llvm-project/build")
os.environ["LLVM_BUILD_DIR"] = llvm_build_dir

# source ~/xilinx_vitis.sh
# source /opt/xilinx/xrt/setup.sh

num_particles = 8
num_feats = 3
# Smallest model for transformer
num_transformers = 4
embbed_dim = 8
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

model = ConstituentNet(**model_config).eval()
example_inputs = [torch.rand(batch_size, num_particles, num_feats)]
golden = model(*example_inputs)
np_input = example_inputs[0].detach().numpy()

# LLVM
llvm_mod = allo.frontend.from_pytorch(
    model,
    example_inputs=example_inputs,
    leaf_modules=(SliceClsToken,),
    verbose=False,
)
res = llvm_mod(np_input)
np.testing.assert_allclose(res, golden.detach().numpy(), atol=1e-5)
print("Test passed!")


# VITIS HLS
# mode = "hw_emu"
# os.environ["XDEVICE"] = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
# os.environ["XCL_EMULATION_MODE"] = mode

# vitis_mod = allo.frontend.from_pytorch(
#     model,
#     example_inputs=example_inputs,
#     leaf_modules=(SliceClsToken,),
#     target="vitis_hls",
#     mode=mode,
#     project="model0_hw.prj",
#     verbose=False,
# )

# allo_out = np.zeros((batch_size, 5), dtype=np.float32)
# vitis_mod(np_input, allo_out)
# np.testing.assert_allclose(allo_out, golden.detach().numpy(), rtol=1e-5, atol=1e-5)
# print("Passed!")
