import subprocess
import time

def monitor_gpu_and_save(interval, output_file):
    with open(output_file, "w") as f:  # "w" 모드로 파일 열기
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
    interval = 0.001  # 관측 간격(초)
    output_file = "gpu_status.txt"  # 저장할 파일 이름
    monitor_gpu_and_save(interval, output_file)
