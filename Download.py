from modelscope import snapshot_download
import os

# 要下载的模型列表,按需取消注释
model_ids = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

# 缓存目录（用于存放中间文件，避免重复下载）
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

for model_id in model_ids:
    model_name = model_id.split('/')[-1]
    local_dir = f"./Qwen/{model_name}"

    print(f"正在从 ModelScope 下载 {model_id} 到 {local_dir}...")
    snapshot_download(
        model_id,
        local_dir=local_dir,
        cache_dir=cache_dir
    )
    print(f"下载完成！模型保存在 {local_dir}\n")