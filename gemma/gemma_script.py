# title       : gemma_run.py
# description : iteration script with example_chat_completion.py of gemma
# author      : Kim Seong Ho
# email       : klue980@gmail.com 
# since       : 2024.04.25
# update      : 2024.04.25

# gemma_run.py : iteration script with example_chat_completion.py of gemma
# use subprocess.run to repeatedly put CLI commands

import subprocess

initial_setting = """
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget vim && \
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com && \
git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
cd /workspace/TensorRT-LLM/examples/gemma && \
pip3 install -r requirements.txt && \
python3 -c "import torch; print(torch.__version__)" && \
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"  && \
python3 -c "import mpmath; print(mpmath.__version__)"  && \
python3 -c "import jax; print(jax.__version__)"  && \
git clone https://github.com/seongho-git/Lab-TensorRT-LLM.git && \
git config --global user.email klue980@gmail.com && git config --global user.name seongho-git && \
cd /workspace/TensorRT-LLM/examples/gemma/Lab-TensorRT-LLM && \
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsightsystems-linux-public-2024.1.1.59-3380207.run && \
bash nsightsystems-linux-public-2024.1.1.59-3380207.run
"""

setting = """
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget vim && \
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com && \
cd /workspace/TensorRT-LLM/examples/gemma && \
pip3 install -r requirements.txt && \
python3 -c "import torch; print(torch.__version__)" && \
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"  && \
python3 -c "import mpmath; print(mpmath.__version__)"  && \
python3 -c "import jax; print(jax.__version__)"  && \
git config --global user.email klue980@gmail.com && git config --global user.name seongho-git && \
cd /workspace/TensorRT-LLM/examples/gemma/Lab-TensorRT-LLM && \
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsightsystems-linux-public-2024.1.1.59-3380207.run && \
bash nsightsystems-linux-public-2024.1.1.59-3380207.run
"""
fastsetting = """
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget vim && \
cd /workspace/TensorRT-LLM/examples/gemma && \
pip3 install -r requirements.txt && \
python3 -c "import torch; print(torch.__version__)" && \
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"  && \
python3 -c "import mpmath; print(mpmath.__version__)"  && \
python3 -c "import jax; print(jax.__version__)"  && \
git config --global user.email klue980@gmail.com && git config --global user.name seongho-git && \
cd /workspace/TensorRT-LLM/examples/gemma/Lab_TensorRT-LLM && \
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsightsystems-linux-public-2024.1.1.59-3380207.run && \
bash nsightsystems-linux-public-2024.1.1.59-3380207.run
"""

export PATH="$PATH:/opt/nvidia/nsight-systems/2024.1.1/bin"
source ~/.bashrc

# setting
part_hf_setting = f"""
mkdir -p ./check/hf/2b/bf16 && \
    mkdir -p ./trt-engine/hf/2b/bf16 && \
    mkdir -p ./trt-engine/hf/2b-context-disable/bf16 && \
    mkdir -p ./check/hf/7b/bf16 && \
    mkdir -p ./trt-engine/hf/7b/bf16 && \
    mkdir -p ./trt-engine/hf/7b-context-disable/bf16 && \
    mkdir -p ./NSYS && \
    mkdir -p ./NCU && \
    mkdir -p ./TXT
"""

# under SM80, bf is not working
part_hf_convert = f"""
python3 ./convert_checkpoint.py \
    --ckpt-type hf \
    --model-dir ./gemma-2b \
    --dtype bfloat16 \
    --world-size 1 \
    --output-model-dir ./check/hf/2b/bf16
"""
part_hf_convert7 = f"""
python3 ./convert_checkpoint.py \
    --ckpt-type hf \
    --model-dir ./gemma-7b \
    --dtype bfloat16 \
    --world-size 1 \
    --output-model-dir ./check/hf/7b/bf16
"""

part_build = f"""
trtllm-build --checkpoint_dir ./check/hf/2b/bf16 \
             --gemm_plugin bfloat16 \
             --gpt_attention_plugin bfloat16 \
             --max_batch_size 8 \
             --max_input_len 64 \
             --max_output_len 1024 \
             --lookup_plugin bfloat16 \
             --output_dir ./trt-engine/hf/2b/bf16
"""
part_build7 = f"""
trtllm-build --checkpoint_dir ./check/hf/7b/bf16 \
             --gemm_plugin bfloat16 \
             --gpt_attention_plugin bfloat16 \
             --max_batch_size 8 \
             --max_input_len 64 \
             --max_output_len 1024 \
             --lookup_plugin bfloat16 \
             --output_dir ./trt-engine/hf/7b/bf16
"""
part_unbuild = f"""
trtllm-build --checkpoint_dir ./check/hf/2b/bf16 \
             --gemm_plugin bfloat16 \
             --gpt_attention_plugin bfloat16 \
             --max_batch_size 1 \
             --max_input_len 32768 \
             --max_output_len 32768 \
             --context_fmha disable \
             --output_dir ./trt-engine/hf/2b-context-disable/bf16
"""

part_summarize = f"""
python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir ./gemma-7b \
                        --data_type bf16 \
                        --engine_dir ./trt-engine/hf/7b/bf16 \
                        --batch_size 8 \
                        --max_input_length 64 \
                        --output_len 1024 \
                        --max_ite 1 && \
                            
nsys profile --wait all -t cuda,nvtx,cudnn,cublas -f true --stats true -w true -o ./NSYS/gemma.nsys-rep \
                        python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir ./gemma-2b \
                        --data_type bf16 \
                        --engine_dir ./trt-engine/hf/2b/bf16 \
                        --batch_size 1 \
                        --max_input_length 32768 \
                        --output_len 32768 \
                        --max_ite 1
"""

part_nsight = f"""
ncu --set full --target-processes all -f \
-o ./NCU/gemma.ncu-rep \
 --kernel-name Kernel --launch-skip 7181 --launch-count 1 "python" ../summarize.py --test_trt_llm --max_ite 1 --hf_model_dir ./llama2/7b-chat-hf --data_type fp16 --engine_dir ./llama2/7b-chat-hf/trt_ckpt/fp16/1-gpu/ --max_input_len 256 --output_len 1024 --batch_size 16
"""


# GPU data
[04/29/2024-18:17:25] [TRT-LLM] [W] When enable_context_fmha is turned on, max_num_tokens (8) should be at least tokens_per_block (128), specifying to tokens_per_block (128). At this time, you also need to enable context chunking at runtime, otherwise you may encounter errors.
[04/29/2024-18:17:25] [TRT-LLM] [I] Compute capability: (8, 9)
[04/29/2024-18:17:25] [TRT-LLM] [I] SM count: 128
[04/29/2024-18:17:25] [TRT-LLM] [I] SM clock: 3135 MHz
[04/29/2024-18:17:25] [TRT-LLM] [I] int4 TFLOPS: 821
[04/29/2024-18:17:25] [TRT-LLM] [I] int8 TFLOPS: 410
[04/29/2024-18:17:25] [TRT-LLM] [I] fp8 TFLOPS: 410
[04/29/2024-18:17:25] [TRT-LLM] [I] float16 TFLOPS: 205
[04/29/2024-18:17:25] [TRT-LLM] [I] bfloat16 TFLOPS: 205
[04/29/2024-18:17:25] [TRT-LLM] [I] float32 TFLOPS: 102
[04/29/2024-18:17:25] [TRT-LLM] [I] Total Memory: 23 GiB
[04/29/2024-18:17:25] [TRT-LLM] [I] Memory clock: 10501 MHz
[04/29/2024-18:17:25] [TRT-LLM] [I] Memory bus width: 384
[04/29/2024-18:17:25] [TRT-LLM] [I] Memory bandwidth: 1008 GB/s
[04/29/2024-18:17:25] [TRT-LLM] [I] PCIe speed: 2500 Mbps
[04/29/2024-18:17:25] [TRT-LLM] [I] PCIe link width: 8
[04/29/2024-18:17:25] [TRT-LLM] [I] PCIe bandwidth: 2 GB/s
[04/29/2024-18:17:25] [TRT] [I] [MemUsageChange] Init CUDA: CPU +13, GPU +0, now: CPU 424, GPU 391 (MiB)
[04/29/2024-18:17:28] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1809, GPU +316, now: CPU 2369, GPU 707 (MiB)
