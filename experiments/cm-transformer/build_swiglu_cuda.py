import torch
from torch.utils.cpp_extension import load
m = load(name="swiglu_cuda", sources=["swiglu_cuda.cu"],
         extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120,code=sm_120", "-DGLOG_USE_GLOG_EXPORT"], extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
         build_directory="results/build", verbose=False)
print("built", m.swiglu_coupled(torch.zeros(1024, device="cuda"),
                               torch.zeros(3584, 1024, device="cuda"),
                               torch.zeros(3584, 1024, device="cuda"),
                               torch.zeros(1024, 3584, device="cuda"),
                               torch.zeros(3584, device="cuda"),
                               torch.zeros(112, dtype=torch.int32, device="cuda"),
                               torch.zeros(1, dtype=torch.int32, device="cuda"),
                               torch.zeros(3584, device="cuda"),
                               torch.zeros(1024, device="cuda"), 0.2))
