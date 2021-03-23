import subprocess

def get_cuda_version():
    cuda_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ")  if s.startswith("release")][0].split(' ')[-1]
    return cuda_version