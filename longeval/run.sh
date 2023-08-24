#CUDA_VISIBLE_DEVICES=7 python3 eval.py --model-name-or-path /data1/csw_model_weights/OriginOne-30b-v1.8.6/vicuna --task lines --test_dir /data0/csw/LongChat

CUDA_VISIBLE_DEVICES=1 python3 eval.py --model-name-or-path /data1/csw_model_weights/Llama-2-13b-chat-hf --task lines --test_dir /data0/csw/LongChat

CUDA_VISIBLE_DEVICES=1 python3 eval.py --model-name-or-path /data1/csw_model_weights/Llama-2-7b-chat-hf --task lines --test_dir /data0/csw/LongChat

CUDA_VISIBLE_DEVICES=1 python3 eval.py --model-name-or-path /data1/csw_model_weights/vicuna-13b-v1.3 --task lines --test_dir /data0/csw/LongChat

CUDA_VISIBLE_DEVICES=1 python3 eval.py --model-name-or-path /data1/csw_model_weights/vicuna-7b-v1.3 --task lines --test_dir /data0/csw/LongChat
