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
    output_file = "2>&1 | tee ./nvTXT/gemma2ite10ba64in64out32_status.txt"  # file name
    monitor_gpu_and_save(output_file)

'''
nvidia-smi --query-gpu=utilization.gpu,memory.used,name \
    --format=csv -lms 1 \
    | tee ./nvTXT/gemma2ite10ba64in64out32_status.csv
'''