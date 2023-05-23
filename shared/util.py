import torch
import os
import torchaudio
import yaml
from shared.default_config import DEFAULT_CONFIG


def load_config_from_yaml():
    if not os.path.exists("config.yaml"):
        with open("config.yaml", "w") as f:
            yaml.dump(DEFAULT_CONFIG, f)

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def register_config_value(key, value):
    config = load_config_from_yaml()
    if not key in config:
        config[key] = value
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)


def get_recc_device(gpu_str=""):
    "source: github.com/drscotthawley/aeiou.core: utility to suggest which pytorch device to use"
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda" if gpu_str == "" else f"cuda:{gpu_str}"
    elif (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):  # must check for mps attr if using older pytorch
        device_str = "mps"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return device_str


def save_audio(audio_out, output_path: str, sample_rate, id_str: str = None):
    files = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(
            output_path,
            f"sample_{id_str}_{ix + 1}.wav"
            if (id_str != None)
            else f"sample_{ix + 1}.wav",
        )
        open(output_file, "a").close()
        output = sample.cpu()

        torchaudio.save(output_file, output, int(sample_rate))
        files.append(output_file)
    return files
