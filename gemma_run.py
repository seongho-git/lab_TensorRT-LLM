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
# --batch_size 128 --max_input_len 512 --output_len 2048
max_ite = 4 # if hf : 1, trt :1
list_batch_size = [1, 4, 16, 64, 256]
list_max_input_len = [256]
max_input_len = 256
list_output_len = [2, 16, , 1024]

# iteration script
# --test_trt_llm --test_hf
# change 3 metrics
for output_len in list_output_len:
    for batch_size in list_batch_size:
        ex_name = f"ite{max_ite}ba{batch_size}in{max_input_len}out{output_len}"
        base_command = f"nsys profile --wait all -t cuda,nvtx,cudnn,cublas -f true \
                        --stats true -w true -o ./NSYS/{ex_name}.nsys-rep \
                        python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir ./gemma-2b \
                        --data_type bf16 \
                        --engine_dir ./trt-engine/hf/2b/bf16 \
                        --batch_size {batch_size} \
                        --max_input_length {max_input_len} \
                        --output_len {output_len} \
                        --max_ite {max_ite}"
        sed_command = f"2>&1 | tee ./TXT/{ex_name}.txt" # | sed -n '/Output/,$p'
        command = f"{base_command} --batch_size {batch_size} --max_input_len {max_input_len} --output_len {output_len} {sed_command}"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"error : {e}")