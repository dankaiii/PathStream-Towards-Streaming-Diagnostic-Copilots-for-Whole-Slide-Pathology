# PathStream

[arXiv](https://arxiv.org/abs/2611.30167) | [Hugging Face Dataset](https://huggingface.co/datasets/DK-NOTAVAILABLE/PathStream-TCGA-Trajectories) | [Cite](#citation)

PathStream is a streaming vision-language model that follows a pathologist as they navigate a whole-slide image. It decides when to describe a finding and uses observations from across the session to generate a final diagnosis.

## Dataset

The [PathStream TCGA Trajectories dataset on Hugging Face](https://huggingface.co/datasets/DK-NOTAVAILABLE/PathStream-TCGA-Trajectories) contains 1,791 slide-reading trajectories annotated by five pathologists. The training split (`train/`) contains 1,388 trajectories, and the test split (`test/`) contains 403. Each JSONL file records how one pathologist navigated a single TCGA whole-slide image during one reading session. The first line contains slide metadata, microscopic findings, and the pathological diagnosis. Each subsequent line records the viewport position, size, center, and zoom for one frame.

## Repository guide

| Directory | Contents |
| --- | --- |
| [data/](data/) | Loads training records, matches report findings to frames, and caches visual features from slide viewports. |
| [examples/](examples/) | An example reading session with its frames and [sample training records](examples/train.jsonl). |
| [model/](model/) | PathStream’s Qwen3-VL backend and streaming inference code, plus support for LiveCC and MM-Duet2. |
| [losses/](losses/) | Training objectives for response timing, descriptions, and diagnosis, including attention supervision over diagnostic evidence. |
| [train/](train/) | Two-stage PathStream training, validation, and domain fine-tuning for baseline models. |
| [configs/](configs/) | Default training settings and ablation presets. |
| [benchmark/](benchmark/README.md) | Build inputs, run models, and score the four PathStreamBench tasks covering response timing, streaming morphological description, session-level diagnosis, and pathology video understanding. |

## Training

Use Linux with Python 3.11, CUDA, `tmux`, and CUDA-enabled PyTorch 2.4.0 and torchvision 0.19.0. From the repository root, install the dependencies.

```bash
python -m pip install setuptools wheel packaging ninja
python -m pip install --no-build-isolation -r requirements.txt
```

Set the model, data, feature, and output paths in [configs/train.yaml](configs/train.yaml). Both training stages read `train.jsonl` and `val.jsonl` from the configured data directory. Run both stages with the following command.

```bash
bash train.sh
```

Use [ablate.sh](ablate.sh) to run a preset from [configs/ablations.yaml](configs/ablations.yaml), or [domain_sft.sh](domain_sft.sh) to fine-tune LiveCC or MM-Duet2.

## Citation

```bibtex
@misc{liao2026pathstreamstreamingdiagnosticcopilots,
  title         = {{PathStream}: Towards Streaming Diagnostic Copilots for Whole-Slide Pathology},
  author        = {Dankai Liao and YuFeng Wu and Ziyue Wang and Guanxi Liu and Tianyi Zhang and Zeyu Liu and Linghan Cai and Yueming Jin},
  year          = {2026},
  eprint        = {2611.30167},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2611.30167}
}
```
