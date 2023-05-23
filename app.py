from importlib import import_module
import gradio as gr
import os
import torch

# Check cuda

if torch.cuda.is_available():
    print("Using CUDA")


# load_modules
def load_modules():
    modules = []
    for folder in os.listdir("modules"):
        for file in os.listdir(os.path.join("modules", folder)):
            if file.endswith(".py") and file.startswith("ext_"):
                extfile = f'modules.{folder}.{file.split(".")[0]}'
                modules.append(extfile)
    modules.sort(key=lambda x: x.split(".")[2] == "Inference", reverse=True)
    for module in modules:
        import_module(module, package=None)
        print(f"Loaded module: {module}")


# load css
with open("styles.css", "r") as css_file:
    css = css_file.read()

# launch
with gr.Blocks(title="Sample Diffusion WebUI", css=css) as demo:
    load_modules()


# ****************************************************************************
demo.queue(concurrency_count=2, api_open=False)
demo.launch(
    show_api=True,
)
