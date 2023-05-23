import os
import wandb
import subprocess
from threading import Thread
import gradio as gr
from modules.Trainer.model_map import model_map
from urllib.parse import urlparse
import hashlib
import urllib.request
from tqdm import tqdm
import traceback


trainproc = None


def run_subp(args):
    global trainproc
    # Start the process using subprocess.Popen
    trainproc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        while trainproc.poll() is None:
            procoutput = trainproc.stdout.readline().decode()
            if procoutput:
                print(procoutput.strip())
    except:
        pass


def stop_subp(proc):
    global trainproc
    if trainproc is not None:
        trainproc.terminate()
        print("Stopping training..")
        trainproc = None
        return gr.Button.update(interactive=True)
    return


def run_training(
    ckpt_path,
    name,
    training_dir,
    training_zip,
    sample_size,
    accum_batches,
    sample_rate,
    batch_size,
    demo_every,
    checkpoint_every,
    num_workers,
    num_nodes,
    num_gpus,
    random_crop,
    max_epochs,
    save_path,
    wandb_key,
):
    print("Starting training..")
    if save_path == "training_checkpoints" and not os.path.exists(
        "training_checkpoints"
    ):
        os.mkdir("training_checkpoints")

    args = [
        "venv\\Scripts\\python",
        "extensions\\dd_trainer\\train_uncond.py",
        "--ckpt-path",
        ckpt_path,
        "--name",
        name,
        "--training-dir",
        training_dir,
        "--training-zip",
        training_zip,
        "--sample-size",
        sample_size,
        "--accum-batches",
        accum_batches,
        "--sample-rate",
        sample_rate,
        "--batch-size",
        batch_size,
        "--demo-every",
        demo_every,
        "--checkpoint-every",
        checkpoint_every,
        "--num-workers",
        num_workers,
        "--num-nodes",
        num_nodes,
        "--num-gpus",
        num_gpus,
        "--random-crop",
        f"True" if random_crop else "",
        "--max-epochs",
        max_epochs,
        "--save-path",
        f"{save_path}/{name}",
    ]

    wandb.login(key=wandb_key)
    yield gr.Button.update(interactive=False)
    t = Thread(target=run_subp, args=args, daemon=True)
    t.start()
    t.join()
    return gr.Button.update(interactive=True)


def download_model(diffusion_model_name, uri_index=0):
    if diffusion_model_name != "custom":
        model_filename = get_model_filename(diffusion_model_name)
        model_local_path = os.path.join("base_models", model_filename)
        if not os.path.exists("base_models"):
            os.makedirs("base_models")

        if os.path.exists(model_local_path):
            print(
                f"{diffusion_model_name} already downloaded. If the file is corrupt, enable check_model_SHA."
            )

        else:
            for model_uri in model_map[diffusion_model_name]["uri_list"]:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
                        "Referer": "https://www.example.com/",
                    }
                    opener = urllib.request.build_opener()
                    opener.addheaders = headers.items()
                    urllib.request.install_opener(opener)

                    with tqdm(
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        miniters=1,
                        desc=model_filename,
                    ) as t:
                        urllib.request.urlretrieve(
                            model_uri,
                            model_local_path,
                            reporthook=lambda blocknum, blocksize, totalsize: t.update(
                                blocknum * blocksize - t.n
                            ),
                        )

                    with open(model_local_path, "rb") as f:
                        bytes = f.read()
                        hash = hashlib.sha256(bytes).hexdigest()
                        print(f"SHA: {hash}")

                    if os.path.exists(model_local_path):
                        model_map[diffusion_model_name]["downloaded"] = True
                        return
                    else:
                        print(
                            f"{diffusion_model_name} model download from {model_uri} failed. Will try any fallback uri."
                        )
                except Exception as e:
                    print(
                        f"{diffusion_model_name} download failed with exception: {str(e)}"
                    )

            print(f"{diffusion_model_name} download failed.")


def get_model_filename(diffusion_model_name):
    model_uri = model_map[diffusion_model_name]["uri_list"][0]
    model_filename = os.path.basename(urlparse(model_uri).path)
    return model_filename


def swap_base_model(base_model_select, progress=gr.Progress(track_tqdm=True)):
    if base_model_select == "Custom":
        return [gr.Dropdown.update(), gr.File.update(visible=True)]
    else:
        download_model(base_model_select)
        try:
            selected_model = os.path.join(
                "base_models", get_model_filename(base_model_select)
            )
        except:
            selected_model = None
        return [gr.Dropdown.update(), gr.File.update(value=selected_model)]


def swap_training_mode(training_mode):
    if training_mode == "Directory":
        return [gr.Textbox.update(visible=True), gr.File.update(visible=False)]
    elif training_mode == "ZIP File":
        return [gr.Textbox.update(visible=False), gr.File.update(visible=True)]


with gr.Tab("Trainer") as UI:
    name = gr.Textbox(label="Model Name", value="MyModel")

    with gr.Row():
        base_model_select = gr.Dropdown(
            label="Base Model",
            choices=list(model_map.keys()) + ["Custom"],
            value="Custom",
        )
        ckpt_path = gr.File(label="Base Model (.ckpt)", visible=True)
    with gr.Row():
        training_mode = gr.Dropdown(
            label="Training Data Type",
            choices=["Directory", "ZIP File"],
            value="Directory",
        )
        training_dir = gr.Textbox(label="Training Directory")
        training_zip = gr.File(label="Training ZIP", visible=False)
    sample_size = gr.Slider(
        label="Chunk Size", value=65536, minimum=32768, maximum=524288, step=32768
    )
    accum_batches = gr.Number(label="Accumulate Batches", value=2)
    sample_rate = gr.Number(label="Sample Rate", value=48000)
    batch_size = gr.Number(label="Batch Size", value=1)
    random_crop = gr.Checkbox(label="Random Crop", value=False)
    wandb_key = gr.Textbox(label="WandB Key", type="password")
    with gr.Accordion("Advanced", open=False):
        demo_every = gr.Number(label="Demo Every *** Steps", value=250)
        checkpoint_every = gr.Number(label="Export Model Every *** Steps", value=500)
        num_workers = gr.Number(label="Number Of Workers", value=1)
        num_nodes = gr.Number(label="Number Of Nodes", value=1)
        num_gpus = gr.Number(label="Number Of GPUs", value=1)
        max_epochs = gr.Number(label="Max Epochs", value=10000000)
        save_path = gr.Textbox(label="Save Path", value="training_checkpoints")

    training_mode.change(
        fn=swap_training_mode,
        inputs=[training_mode],
        outputs=[training_dir, training_zip],
    )

    base_model_select.change(
        fn=swap_base_model,
        inputs=[base_model_select],
        outputs=[base_model_select, ckpt_path],
    )
    with gr.Row():
        start_btn = gr.Button("Start")
        stop_btn = gr.Button("Stop")
    start_btn.click(
        fn=run_training,
        inputs=[
            ckpt_path,
            name,
            training_dir,
            training_zip,
            sample_size,
            accum_batches,
            sample_rate,
            batch_size,
            demo_every,
            checkpoint_every,
            num_workers,
            num_nodes,
            num_gpus,
            random_crop,
            max_epochs,
            save_path,
            wandb_key,
        ],
        outputs=[],
    )
