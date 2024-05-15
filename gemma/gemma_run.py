# title       : gemma_run.py
# description : iteration script with example_chat_completion.py of gemma
# author      : Kim Seong Ho
# email       : klue980@gmail.com 
# since       : 2024.04.25
# update      : 2024.04.25

# gemma_run.py : iteration script with example_chat_completion.py of gemma
# use subprocess.run to repeatedly put CLI commands

import subprocess

# sweep parameter
# --batch_size 512 --max_input_len 64 --output_len 512
max_ite = 2 # if hf : 1, trt :1
list_batch_size = [8] # [1, 8]
# batch_size = 1
list_max_input_len = [64] # [1, 4, 16, 64, 256]
# max_input_len = 512
list_output_len = [512] # [1, 4, 16, 64, 256, 1024]

# iteration script
# --test_trt_llm --test_hf
# change 3 metrics
for batch_size in list_batch_size:
    for max_input_len in list_max_input_len: 
        build_command = f"trtllm-build --checkpoint_dir /workspace/TensorRT-LLM/examples/gemma/check/hf/2b/bf16 \
                            --gemm_plugin bfloat16 \
                            --gpt_attention_plugin bfloat16 \
                            --max_batch_size {batch_size} \
                            --max_input_len {max_input_len} \
                            --max_output_len 1024 \
                            --lookup_plugin bfloat16 \
                            --output_dir /workspace/TensorRT-LLM/examples/gemma/trt-engine/hf/2b/bf16"
        try:
            print(build_command)
            subprocess.run(build_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"error : {e}")
        for output_len in list_output_len:
            ex_name = f"gemma2bite{max_ite}ba{batch_size}in{max_input_len}out{output_len}"
            base_command = f"nsys profile --wait all -t cuda,nvtx,cudnn,cublas -f true \
                            --stats true -w true -o /workspace/TensorRT-LLM/examples/gemma/NSYS/{ex_name}.nsys-rep \
                            python3 /workspace/TensorRT-LLM/examples/summarize.py --test_trt_llm \
                            --hf_model_dir /workspace/TensorRT-LLM/examples/gemma/gemma-2b \
                            --data_type bf16 \
                            --engine_dir /workspace/TensorRT-LLM/examples/gemma/trt-engine/hf/2b/bf16 \
                            --batch_size {batch_size} \
                            --max_input_length {max_input_len} \
                            --output_len {output_len} \
                            --max_ite {max_ite}"
            sed_command = f"2>&1 | tee /workspace/TensorRT-LLM/examples/gemma/TXT/{ex_name}.txt" # | sed -n '/Output/,$p'
            command = f"{base_command} {sed_command}"
            try:
                print(command)
                subprocess.run(command, shell=True, check=True)
                # subprocess.run(command, shell=True, check=True) # if needed
            except subprocess.CalledProcessError as e:
                print(f"error : {e}")
                
"""
python3 /workspace/TensorRT-LLM/examples/summarize.py --test_trt_llm --data_type bf16 \
--hf_model_dir /workspace/TensorRT-LLM/examples/llama/Llama-2-7b-hf \
--engine_dir /workspace/TensorRT-LLM/examples/llama/trt-engine/hf/2b/bf16 \
--batch_size 64 --max_input_length 64 --output_len 2 --max_ite 10 \
2>&1 | tee /workspace/TensorRT-LLM/examples/llama/TXT/llama2ite10ba64in64out2.txt
"""
"""
python3 /workspace/TensorRT-LLM/examples/summarize.py --test_trt_llm --data_type bf16 \
--hf_model_dir /workspace/TensorRT-LLM/examples/llama/Llama-2-7b-hf \
--engine_dir /workspace/TensorRT-LLM/examples/llama/trt-engine/hf/2b/bf16 \
--max_input_length 64 --max_ite 10 --batch_size 1 --output_len 2
"""
"""
python3 /workspace/TensorRT-LLM/examples/summarize.py --test_trt_llm --data_type bf16 \
--hf_model_dir /workspace/TensorRT-LLM/examples/llama/Meta-Llama-3-8B \
--engine_dir /workspace/TensorRT-LLM/examples/llama/trt-engine/hf/3-8b/bf16 \
--max_input_length 64 --max_ite 10 --batch_size 1 --output_len 2
"""