import os
from shared.const import MODEL_FOLDER
from modules.Inference.const import MAX_AUDIOBOXES
import gradio as gr


def load_models():
    list_of_files = []
    for dirpath, dirnames, filenames in os.walk(MODEL_FOLDER):
        for filename in filenames:
            if filename.endswith(".ckpt"):
                list_of_files.append(
                    os.sep.join([dirpath.replace(MODEL_FOLDER, ""), filename])[1:]
                )
    return list_of_files


def refresh_all_models(*inputs):
    models = load_models()
    selected = models[0]
    return gr.Dropdown.update(value=selected, choices=models)


def make_audio_outputs(amount):
    audio_outputs = []
    for i in range(amount):
        audio_outputs.append(gr.components.Audio(label=f"Batch #{i+1}"))
    return audio_outputs


def variable_outputs(output_amt, mode, interp_amount):
    output_amt = (
        int(output_amt)
        if (mode != "Interpolation")
        else int(output_amt) * int(interp_amount)
    )

    return [gr.Audio.update(visible=True)] * output_amt + [
        gr.Audio.update(visible=False)
    ] * (MAX_AUDIOBOXES - output_amt)
