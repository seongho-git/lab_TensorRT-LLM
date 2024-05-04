# title       : llama_run.py
# description : iteration script with example_chat_completion.py of llama
# author      : Kim Seong Ho
# email       : klue980@gmail.com 
# since       : 2024.04.25
# update      : 2024.04.25

# llama_run.py : iteration script with example_chat_completion.py of llama
# use subprocess.run to repeatedly put CLI commands

setting = """
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget vim && \
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com && \
cd /workspace/TensorRT-LLM/examples/llama && \
pip3 install -r requirements.txt && \
python3 -c "import torch; print(torch.__version__)" && \
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"  && \
python3 -c "import mpmath; print(mpmath.__version__)"  && \
git config --global user.email klue980@gmail.com && git config --global user.name seongho-git && \
cd /workspace/TensorRT-LLM/examples/llama/Lab_TensorRT-LLM && \
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsightsystems-linux-public-2024.1.1.59-3380207.run && \
bash nsightsystems-linux-public-2024.1.1.59-3380207.run
"""
fastsetting = """
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget vim && \
cd /workspace/TensorRT-LLM/examples/llama && \
pip3 install -r requirements.txt && \
python3 -c "import torch; print(torch.__version__)" && \
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"  && \
python3 -c "import mpmath; print(mpmath.__version__)"  && \
git config --global user.email klue980@gmail.com && git config --global user.name seongho-git && \
cd /workspace/TensorRT-LLM/examples/llama/Lab_TensorRT-LLM && \
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsightsystems-linux-public-2024.1.1.59-3380207.run && \
bash nsightsystems-linux-public-2024.1.1.59-3380207.run
"""

export PATH="$PATH:/opt/nvidia/nsight-systems/2024.1.1/bin"
source ~/.bashrc

# setting
part_hf_setting = f"""
mkdir -p ./check/hf/3-8b/bf16 && \
    mkdir -p ./trt-engine/hf/3-8b/bf16 && \
    mkdir -p ./trt-engine/hf/3-8b-context-disable/bf16 && \
    mkdir -p ./NSYS && \
    mkdir -p ./NCU && \
    mkdir -p ./TXT
"""

# under SM80, bf is not working
part_hf_convert = f"""
python3 ./convert_checkpoint.py \
    --model_dir ./Meta-Llama-3-8B \
    --dtype bfloat16 \
    --output_dir ./check/hf/3-8b/bf16
"""

part_build = f"""
trtllm-build --checkpoint_dir ./check/hf/3-8b/bf16 \
             --gemm_plugin bfloat16 \
             --gpt_attention_plugin bfloat16 \
             --max_batch_size 64 \
             --max_input_len 1024 \
             --max_output_len 1024 \
             --lookup_plugin bfloat16 \
             --output_dir ./trt-engine/hf/3-8b/bf16
"""

part_unbuild = f"""
trtllm-build --checkpoint_dir ./check/hf/3-8b/bf16 \
             --gemm_plugin bfloat16 \
             --gpt_attention_plugin bfloat16 \
             --max_batch_size 1 \
             --max_input_len 32768 \
             --max_output_len 32768 \
             --context_fmha disable \
             --output_dir ./trt-engine/hf/3-8b-context-disable/bf16
"""

part_summarize = f"""
python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir ./Meta-Llama-3-8B \
                        --data_type bf16 \
                        --engine_dir ./trt-engine/hf/3-8b/bf16 \
                        --batch_size 64 \
                        --max_input_length 1024 \
                        --output_len 1024 \
                        --max_ite 1 && \
                            
nsys profile --wait all -t cuda,nvtx,cudnn,cublas -f true --stats true -w true -o ./NSYS/llama.nsys-rep \
                        python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir ./Meta-Llama-3-8B \
                        --data_type bf16 \
                        --engine_dir ./trt-engine/hf/3-8b/bf16 \
                        --batch_size 1 \
                        --max_input_length 32768 \
                        --output_len 32768 \
                        --max_ite 1
"""