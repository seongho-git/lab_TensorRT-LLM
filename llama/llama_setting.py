# title       : gemma_run/workspace/TensorRT-LLM/examples/llamapy
# description : iteration script with example_chat_completion/workspace/TensorRT-LLM/examples/llamapy of gemma
# author      : Kim Seong Ho
# email       : klue980@gmail/workspace/TensorRT-LLM/examples/llamacom 
# since       : 2024/workspace/TensorRT-LLM/examples/llama04/workspace/TensorRT-LLM/examples/llama25
# update      : 2024/workspace/TensorRT-LLM/examples/llama04/workspace/TensorRT-LLM/examples/llama25

# gemma_run/workspace/TensorRT-LLM/examples/llamapy : iteration script with example_chat_completion/workspace/TensorRT-LLM/examples/llamapy of gemma
# use subprocess/workspace/TensorRT-LLM/examples/llamarun to repeatedly put CLI commands

import subprocess

# sweep parameter
# --batch_size 128 --max_input_len 512 --output_len 2048
max_ite = 1 # if hf : 1, trt :1
list_batch_size = [1, 64, 4096]
# batch_size = 1
# list_max_input_len = [1, 8, 64, 512, 4096, 32768]
max_input_len = 128
list_output_len = [8, 128, 32768]

# iteration script
# --test_trt_llm --test_hf
# change 3 metrics
for output_len in list_output_len:
    for batch_size in list_batch_size:
        ex_name = f"ite{max_ite}ba{batch_size}in{max_input_len}out{output_len}"
        build_command = f"trtllm-build --checkpoint_dir /workspace/TensorRT-LLM/examples/llama/check/hf/2b/bf16 \
                         --gemm_plugin bfloat16 \
                         --gpt_attention_plugin bfloat16 \
                         --max_batch_size {batch_size} \
                         --max_input_len {max_input_len} \
                         --max_output_len {output_len} \
                         --lookup_plugin bfloat16 \
                         --output_dir /workspace/TensorRT-LLM/examples/llama/trt-engine/hf/2b/bf16"
        base_command = f"nsys profile --wait all -t cuda,nvtx,cudnn,cublas -f true \
                        --stats true -w true -o /workspace/TensorRT-LLM/examples/llama/NSYS/{ex_name}/workspace/TensorRT-LLM/examples/llamansys-rep \
                        python3 /workspace/TensorRT-LLM/examples/summarize.py --test_trt_llm \
                        --hf_model_dir /workspace/TensorRT-LLM/examples/llama/gemma-2b \
                        --data_type bf16 \
                        --engine_dir /workspace/TensorRT-LLM/examples/llama/trt-engine/hf/2b/bf16 \
                        --batch_size {batch_size} \
                        --max_input_length {max_input_len} \
                        --output_len {output_len} \
                        --max_ite {max_ite}"
        sed_command = f"2>&1 | tee /workspace/TensorRT-LLM/examples/llama/TXT/{ex_name}/workspace/TensorRT-LLM/examples/llamatxt" # | sed -n '/Output/,$p'
        command = f"{base_command} {sed_command}"
        try:
            print(build_command)
            subprocess/workspace/TensorRT-LLM/examples/llamarun(build_command, shell=True, check=True)
            print(command)
            subprocess/workspace/TensorRT-LLM/examples/llamarun(command, shell=True, check=True)
        except subprocess/workspace/TensorRT-LLM/examples/llamaCalledProcessError as e:
            print(f"error : {e}")