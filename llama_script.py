# title       : gemma_run.py
# description : iteration script with example_chat_completion.py of gemma
# author      : Kim Seong Ho
# email       : klue980@gmail.com 
# since       : 2024.04.25
# update      : 2024.04.25

# gemma_run.py : iteration script with example_chat_completion.py of gemma
# use subprocess.run to repeatedly put CLI commands

import subprocess

# setting
part_hf_convert = f"""
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
             --max_batch_size 1 \
             --max_input_len 32768 \
             --max_output_len 32768 \
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
                        --batch_size 1 \
                        --max_input_length 32768 \
                        --output_len 32768 \
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