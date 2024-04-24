# setting
part_hf_convert = f"""
mkdir -p ./check/hf/2b/bf16 && \
    mkdir -p ./trt-engine/hf/2b/bf16 && \
    mkdir -p ./trt-engine/hf/2b-context-disable/bf16 && \
    mkdir -p ./NSYS && \
    mkdir -p ./NCU
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

part_build = f"""
trtllm-build --checkpoint_dir ./check/hf/2b/bf16 \
             --gemm_plugin bfloat16 \
             --gpt_attention_plugin bfloat16 \
             --max_batch_size 1 \
             --max_input_len 384 \
             --max_output_len 2 \
             --lookup_plugin bfloat16 \
             --output_dir ./trt-engine/hf/2b/bf16
"""

part_unbuild = f"""
trtllm-build --checkpoint_dir ./check/hf/2b/bf16 \
             --gemm_plugin bfloat16 \
             --gpt_attention_plugin bfloat16 \
             --max_batch_size 1 \
             --max_input_len 384 \
             --max_output_len 2 \
             --context_fmha disable \
             --output_dir ./trt-engine/hf/2b-context-disable/bf16
"""

part_summarize = f"""
python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir ./gemma-2b \
                        --data_type bf16 \
                        --engine_dir ./trt-engine/hf/2b/bf16 \
                        --batch_size 1 \
                        --max_input_length 384 \
                        --output_len 2 \
                        --max_ite 1 && \
                            
nsys profile --wait all -t cuda,nvtx,cudnn,cublas -f true --stats true -w true -o ./NSYS/gemmaite1ba1in384o2.nsys-rep \
                        python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir ./gemma-2b \
                        --data_type bf16 \
                        --engine_dir ./trt-engine/hf/2b/bf16 \
                        --batch_size 1 \
                        --max_input_length 384 \
                        --output_len 2 \
                        --max_ite 1
"""