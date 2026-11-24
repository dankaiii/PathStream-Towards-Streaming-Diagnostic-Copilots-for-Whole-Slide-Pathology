"""Portable inference environments. Run from the release directory.

python -m benchmark.run_benchmark.streaming_environments list
python -m benchmark.run_benchmark.streaming_environments install pathstream-streamingvlm
python -m benchmark.run_benchmark.streaming_environments run MODEL TASK --help
The loader's source_dir argument points to your upstream VideoLLM-online, StreamingVLM
or MMDuet2 checkout. Local checkpoint paths and hosted API credentials are caller-supplied.
"""
import argparse
import json
import os
import subprocess

MODEL_ENV = {'aura': 'pathstream-aura',
 'gpt_api': 'pathstream-api',
 'internvl35': 'pathstream-vlm-common',
 'lingshu': 'pathstream-vlm-common',
 'livecc': 'pathstream-vlm-common',
 'livecc_sft': 'pathstream-vlm-common',
 'mmduet2': 'pathstream-mmduet2',
 'mmduet2_sft': 'pathstream-mmduet2',
 'moss_vl_realtime': 'pathstream-moss-vl-realtime',
 'patho_r1': 'pathstream-vlm-common',
 'pathstream': 'pathstream-vlm-common',
 'qwen36': 'pathstream-api',
 'qwen38_flash': 'pathstream-api',
 'qwen3_vl_4b': 'pathstream-vlm-common',
 'qwen3_vl_8b': 'pathstream-vlm-common',
 'streamingvlm': 'pathstream-streamingvlm',
 'videollm_online': 'pathstream-vlm-common'}

ENVIRONMENTS = {'pathstream-aura': {'conda': ['python=3.12', 'pip'],
                     'torch': ['torch==2.10.0', 'torchvision==0.25.0', 'torchaudio==2.10.0'],
                     'index': 'https://download.pytorch.org/whl/cu128',
                     'packages': ['transformers==4.57.6',
                                  'accelerate==1.12.0',
                                  'numpy==2.2.6',
                                  'Pillow==12.1.1',
                                  'safetensors==0.7.0',
                                  'tokenizers==0.22.2',
                                  'huggingface-hub==0.36.2',
                                  'vllm==0.17.1',
                                  'xxhash==3.6.0',
                                  'einops==0.8.2'],
                     'wheels': []},
 'pathstream-mmduet2': {'conda': ['python=3.10', 'pip'],
                        'torch': ['torch==2.4.0', 'torchvision==0.19.0'],
                        'index': 'https://download.pytorch.org/whl/cu121',
                        'packages': ['transformers==4.49.0',
                                     'accelerate==1.5.2',
                                     'numpy==1.26.4',
                                     'Pillow==10.4.0',
                                     'safetensors==0.8.0',
                                     'tokenizers==0.21.2',
                                     'huggingface-hub==0.36.2',
                                     'einops==0.8.1',
                                     'qwen-vl-utils==0.0.8',
                                     'decord==0.6.0',
                                     'six==1.17.0',
                                     'peft==0.17.1'],
                        'wheels': ['https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl']},
 'pathstream-moss-vl-realtime': {'conda': ['python=3.12', 'pip'],
                                 'torch': ['torch==2.8.0', 'torchvision==0.23.0'],
                                 'index': 'https://download.pytorch.org/whl/cu128',
                                 'packages': ['transformers==4.57.1',
                                              'accelerate==1.12.0',
                                              'numpy==2.4.3',
                                              'Pillow==12.1.1',
                                              'safetensors==0.8.0',
                                              'tokenizers==0.22.2',
                                              'huggingface-hub==0.36.2',
                                              'torchcodec==0.7.0',
                                              'joblib==1.5.2',
                                              'einops==0.8.2',
                                              'packaging==26.0'],
                                 'wheels': ['https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl']},
 'pathstream-streamingvlm': {'conda': ['python=3.11', 'pip', 'av=14.4.0'],
                             'torch': ['torch==2.7.1', 'torchvision==0.22.1'],
                             'index': 'https://download.pytorch.org/whl/cu126',
                             'packages': ['transformers==4.52.4',
                                          'accelerate==1.10.1',
                                          'numpy==1.26.4',
                                          'Pillow==10.4.0',
                                          'safetensors==0.8.0',
                                          'tokenizers==0.21.2',
                                          'huggingface-hub==0.36.2',
                                          'qwen-vl-utils==0.0.11',
                                          'einops==0.8.1',
                                          'av==14.4.0'],
                             'wheels': ['https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl']},
 'pathstream-vlm-common': {'conda': ['python=3.11', 'pip'],
                           'torch': ['torch==2.4.0', 'torchvision==0.19.0'],
                           'index': 'https://download.pytorch.org/whl/cu121',
                           'packages': ['transformers==4.57.1',
                                        'accelerate==1.10.1',
                                        'numpy==1.26.4',
                                        'Pillow==10.4.0',
                                        'safetensors==0.8.0',
                                        'tokenizers==0.22.2',
                                        'huggingface-hub==0.36.2',
                                        'einops==0.8.1',
                                        'peft==0.17.1',
                                        'livecc-utils==0.0.2',
                                        'qwen-vl-utils==0.0.11',
                                        'liger-kernel==0.6.4',
                                        'av==12.0.0',
                                        'decord==0.6.0',
                                        'timm==1.0.15',
                                        'sentencepiece==0.2.0'],
                           'wheels': ['https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl']},
 'pathstream-api': {'conda': ['python=3.11', 'pip'], 'packages': ['openai', 'Pillow']}}


def install(conda, name):
    spec = ENVIRONMENTS[name]
    subprocess.run([conda, "create", "-y", "-n", name, "-c", "conda-forge", *spec["conda"]], check=True)
    pip = [conda, "run", "-n", name, "python", "-m", "pip", "install"]
    if "torch" in spec:
        subprocess.run([*pip, *spec["torch"], "--index-url", spec["index"]], check=True)
        subprocess.run([*pip, *spec["packages"]], check=True)
        if spec["wheels"]:
            subprocess.run([*pip, "--no-deps", *spec["wheels"]], check=True)
    else:
        subprocess.run([*pip, *spec["packages"]], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    setup = sub.add_parser("install")
    setup.add_argument("environment", choices=ENVIRONMENTS)
    run = sub.add_parser("run")
    run.add_argument("model", choices=MODEL_ENV)
    run.add_argument("task", choices=("1", "2", "3", "4"))
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    conda = os.environ.get("CONDA_EXE", "conda")
    if args.command == "list":
        print(json.dumps({"models": MODEL_ENV, "environments": ENVIRONMENTS}, indent=2))
    elif args.command == "install":
        install(conda, args.environment)
    else:
        subprocess.run([conda, "run", "--no-capture-output", "-n", MODEL_ENV[args.model],
                        "python", "-m", "benchmark.run_benchmark", args.model, args.task,
                        *args.arguments], check=True)


if __name__ == "__main__":
    main()
