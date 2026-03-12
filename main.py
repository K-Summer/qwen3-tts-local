def main():
    print("Hello from qwen3-tts-api!\n")
    print("运行下面的命令安装依赖:")
    commands = [
        "uv pip install -U qwen3-tts",
        "pip install -e .",
        "uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3%2Bcu130torch2.10-cp314-cp314-win_amd64.whl",
        "uv pip install torch torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu130/",
    ]
    for cmd in commands:
        print(f"  {cmd}")

if __name__ == "__main__":
    main()
