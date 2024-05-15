import subprocess
import time

def monitor_gpu_and_save(output_file):
    base_command = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 1"
    command = f"{base_command} {output_file}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    output_file = "2>&1 | tee ./nvTXT/llama2ite10ba64in64out32_status.txt"  # file name
    monitor_gpu_and_save(output_file)

'''
nvidia-smi --query-gpu=name,compute_cap,memory.total \
    --format=csv \
    | tee ./nvTXT/llama3ite10ba1in64out2_status.csv && \
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv -lms 1 \
    | tee -a ./nvTXT/llama3ite10ba1in64out2_status.csv
'''