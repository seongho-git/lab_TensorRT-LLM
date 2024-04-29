# Lab_TensorRT-LLM
(It is beta version, still making)
* update : 2024.04.29
* script for Llama 3 and Gemma with TensorRT-LLM (NVIDIA) using huggingface model convert version
  
### References
TensorRT-LLM Github  : [TensoRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) \
Original Llama : [Llama & 2](https://github.com/meta-llama/llama) [Llama 3  ](https://github.com/meta-llama/llama3) \
Original Gemma : [Gemma    ](https://github.com/google-deepmind/gemma)

## Prerequisite
### Download TensorRT-LLM
```bash
# Obtain and start the basic docker image environment (optional).
docker run --gpus all -it --privileged --ipc host --name CONTAINER_NAME -v /workspace nvidia/cuda:12.1.0-devel-ubuntu22.04
docker run --gpus all -it --privileged --ipc host --name CONTAINER_NAME -v /workspace runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install dependencies, TensorRT-LLM requires Python 3.10, nvidia/cuda:12.1.0-devel-ubuntu22.04
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget vim
# runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04 version
apt-get update && apt-get -y install openmpi-bin libopenmpi-dev git git-lfs wget vim

# Install the latest preview version (corresponding to the main branch) of TensorRT-LLM.
# If you want to install the stable version (corresponding to the release branch), please
# remove the `--pre` option.
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
pip3 install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com

# download requirements and this repository
git lfs install && git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
cd TensorRT-LLM/examples/gemma && git clone https://github.com/seongho-git/Lab_TensorRT-LLM && \
pip3 install -r requirements.txt

# Check installation
python3 -c "import torch; print(torch.__version__)" && \
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"  && \
python3 -c "import mpmath; print(mpmath.__version__)"  && \
python3 -c "import jax; print(jax.__version__)"
```

### Download Nsight Systems
if needed, Install Nsight Systems
[Nsight Install](https://klue.tistory.com/14)

## Download Models
```bash
# if needed
pip install -U "huggingface_hub[cli]"
git config --global user.email [USER_EMAIL] && git config --global user.name [USER_NAME]
huggingface-cli logout

# download model from huggingface (example)
git lfs install && git lfs clone https://huggingface.co/google/gemma-2b
git lfs install && git lfs clone https://huggingface.co/meta-llama/Meta-Llama-3-8B

# put your username (hf) and write access token (hf-setting)
```

## Using Commands
```bash
# in Lab_TensorRT_LLM folder
python3 gemma_script.py
python3 gemma_run.py
python3 gemma_run_batch.py
python3 llama_script.py
python3 llama_run.py
python3 llama_run_batch.py
```
