import subprocess
import time

def monitor_gpu_and_save(interval, output_file):
    with open(output_file, "w") as f:  # "w" mode for overwriting
        while True:
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv"])
                decoded_output = output.decode("utf-8").strip()
                print(decoded_output)
                f.write(decoded_output + "\n")
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    interval = 0.001  # observation interval (seconds)
    output_file = "./nvTXT/llama2ite10ba1in64out32_status.txt"  # file name
    monitor_gpu_and_save(interval, output_file)
