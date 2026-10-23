"""CPU plumbing smoke of the R6 gold58 worker (10-18, Claude): the kernel's compact_infer 8401812c main() UNCHANGED, with only its
module-level `torch` name proxied so device('cuda') is the CPU (the ci_cpu_launcher.py idea, without its LocalCStudies). fp32, no
--compile (needs CUDA). Same argv as r6g58_run.py otherwise. Validates decode / anchors / side / stackers / flip before GPU time."""
import os, sys
SRC = os.path.expanduser("~/r6g58/src")
sys.path.insert(0, SRC)
import torch
import compact_infer as CI
assert os.path.dirname(os.path.abspath(CI.__file__)) == SRC


class _CpuCuda:
    def __getattr__(self, k):
        return getattr(torch.cuda, k)

    @staticmethod
    def get_device_capability(*a, **k):
        return (7, 0)

    @staticmethod
    def get_device_name(*a, **k):
        return "cpu (smoke)"


class _TorchCpu:
    cuda = _CpuCuda()

    def __getattr__(self, k):
        return getattr(torch, k)

    @staticmethod
    def device(d, *a, **k):
        return torch.device("cpu") if str(d).startswith("cuda") else torch.device(d, *a, **k)


if __name__ == "__main__":
    assert not torch.cuda.is_available(), "hide CUDA"
    assert sys.argv[sys.argv.index("--amp") + 1] == "fp32" and "--compile" not in sys.argv
    CI.torch = _TorchCpu()
    CI.main()
