import gradio as gr
import sys
import torch
import gc

from modules.Inference.const import MAX_AUDIOBOXES
from modules.Inference.util import (
    load_models,
    refresh_all_models,
    variable_outputs,
)


from shared.const import MODEL_FOLDER, REFRESH_SYMBOL, RECC_DEVICE, DEVICE_LIST
from shared.util import save_audio, get_recc_device, register_config_value
from shared.elements import ToolButton, ToolButtonTop, FormRow


sys.path.append("sample_diffusion")
from util.util import load_audio, crop_audio
from util.platform import get_torch_device_type
from dance_diffusion.api import RequestHandler, Request, RequestType, ModelType
from diffusion_library.sampler import SamplerType
from diffusion_library.scheduler import SchedulerType


# ****************************************************************************
# *                                 Generate                                 *
# ****************************************************************************


def generate_audio(
    batch_size,
    model,
    mode,
    use_autocast,
    crop_offset,
    device_accelerator,
    device_offload,
    sample_rate,
    chunk_size,
    seed,
    tame,
    audio_source,
    audio_target,
    mask,
    noise_level,
    interpolations_linear,
    interpolations,
    resamples,
    keep_start,
    steps,
    sigma_max,
    sigma_min,
    rho,
    sampler,
    schedule,
    progress=gr.Progress(track_tqdm=True),
):
    # casting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    request_type = RequestType[mode]
    model_type = ModelType.DD
    sampler_type = SamplerType[sampler]
    scheduler_type = SchedulerType[schedule]
    audio_source = audio_source.name if (audio_source != None) else None
    audio_target = audio_target.name if (audio_target != None) else None
    mask = mask.name if (mask != None) else None

    # load model
    modelpath = f"{MODEL_FOLDER}/{model}"

    device_type_accelerator = (
        device_accelerator if (device_accelerator != None) else get_torch_device_type()
    )
    device_accelerator = torch.device(device_type_accelerator)
    device_offload = torch.device(device_offload)

    request_handler = RequestHandler(
        device_accelerator,
        device_offload,
        optimize_memory_use=False,
        use_autocast=use_autocast,
    )
    seed = (
        int(seed)
        if (seed != -1)
        else torch.randint(0, 4294967294, [1], device=device_type_accelerator).item()
    )

    crop = (
        lambda audio: crop_audio(audio, chunk_size, crop_offset)
        if crop_offset is not None
        else audio
    )
    load_input = (
        lambda source: crop(load_audio(device_accelerator, source, sample_rate))
        if source is not None
        else None
    )

    # make request
    request = Request(
        request_type=request_type,
        model_path=modelpath,
        model_type=model_type,
        model_chunk_size=int(chunk_size),
        model_sample_rate=int(sample_rate),
        seed=int(seed),
        batch_size=int(batch_size),
        audio_source=load_input(audio_source),
        audio_target=load_input(audio_target),
        mask=torch.load(mask) if (mask != None) else None,
        noise_level=noise_level,
        interpolation_positions=interpolations
        if (interpolations_linear == None)
        else torch.linspace(
            0, 1, int(interpolations_linear), device=device_accelerator
        ),
        resamples=int(resamples),
        keep_start=keep_start,
        steps=int(steps),
        sampler_type=sampler_type,
        sampler_args={"use_tqdm": True},
        scheduler_type=scheduler_type,
        scheduler_args={
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "rho": float(rho),
        },
    )

    # process request
    response = request_handler.process_request(request)
    outputs = save_audio(
        (0.5 * response.result).clamp(-1, 1) if (tame == True) else response.result,
        f"Output/{ModelType.DD.__str__()}/{mode.__str__()}/",
        sample_rate,
        f"{seed}",
    )
    outputs += ["data/dummy.mp3"] * (MAX_AUDIOBOXES - len(outputs))
    return outputs

    # ****************************************************************************
    # *                                   MAIN                                   *
    # ****************************************************************************


with gr.Tab("Inference") as UI:
    register_config_value("device_accelerator", RECC_DEVICE)

    with gr.Row():
        with gr.Column():
            models = load_models()
            modes = [
                x for x in RequestType._member_names_ if x != "Inpainting"
            ]  # no inpainting yet
            with gr.Column(variant="panel"):
                with gr.Row():
                    currmodel_comp = gr.components.Dropdown(
                        models, label="Model Checkpoint", value=models[0]
                    )
                    refresh_models = gr.Button(
                        value=REFRESH_SYMBOL,
                        elem_id="model_refresh_button",
                    )
                    refresh_models.click(
                        refresh_all_models, currmodel_comp, currmodel_comp
                    )
                mode_comp = gr.components.Radio(
                    modes, label="Mode of Operation", value="Generation"
                )
                generate_btn = gr.Button(
                    value="Generate Samples", label="Generate", variant="primary"
                )
            with gr.Tab("General Settings"):
                batch_size_comp = gr.components.Slider(
                    label="Batch Size",
                    value=1,
                    maximum=MAX_AUDIOBOXES,
                    minimum=1,
                    step=1,
                )
                gen_components = [
                    gr.components.Checkbox(label="Use Autocast", value=True),
                    gr.components.Number(label="Crop Offset", value=0),
                    gr.components.Radio(
                        DEVICE_LIST, label="Device Accelerator", value=RECC_DEVICE
                    ),
                    gr.components.Radio(
                        DEVICE_LIST, label="Device Offload", value=RECC_DEVICE
                    ),
                    gr.components.Number(label="Sample Rate", value=48000),
                    gr.components.Slider(
                        label="Chunk Size",
                        value=65536,
                        minimum=32768,
                        maximum=524288,
                        step=32768,
                    ),
                    gr.components.Number(label="Seed", value=-1),
                    gr.components.Checkbox(label="Tame", value=True),
                ]
            with gr.Tab("Variation/Interpolation Settings"):
                with gr.Row(variant="panel"):
                    path_components = [
                        gr.File(
                            label="Audio Source Path",
                            interactive=True,
                            file_count="single",
                            file_types=[".mp3", ".wav", ".flac"],
                            elem_id="audio_source_path_file",
                        ),
                        gr.File(
                            label="Audio Target Path",
                            interactive=True,
                            file_count="single",
                            file_types=[".mp3", ".wav", ".flac"],
                            elem_id="audio_target_path_file",
                        ),
                        gr.File(
                            label="Audio Mask Path",
                            interactive=True,
                            file_count="single",
                            file_types=[".mp3", ".wav", ".flac"],
                            elem_id="audio_mask_path_file",
                        ),
                    ]
                noise_level_comp = gr.components.Slider(
                    label="Noise Level", value=0.7, maximum=1, minimum=0
                )
                interpolations_comp = gr.components.Number(
                    label="Interpolations Linear", value=3
                )
                interpolations_pos_comp = gr.components.Textbox(
                    label="Interpolation Positions (comma-separated)", value=None
                )
                resamples_comp = gr.components.Slider(
                    label="Resampling Steps", value=4, maximum=50, minimum=1
                )
                keep_start_comp = gr.components.Checkbox(
                    label="Keep Start of Audio", value=True
                )
                add_components = [
                    noise_level_comp,
                    interpolations_comp,
                    interpolations_pos_comp,
                    resamples_comp,
                    keep_start_comp,
                ]
            with gr.Tab("Sampler Settings"):
                sampler_components = [
                    # extra settings
                    gr.components.Slider(
                        label="Steps", value=50, maximum=250, minimum=10
                    ),
                    gr.components.Number(label="Sigma Min", value=0.1),
                    gr.components.Number(label="Sigma Max", value=50.0),
                    gr.components.Number(label="Rho", value=1.0),
                    gr.components.Radio(
                        SamplerType._member_names_, label="Sampler", value="V_IPLMS"
                    ),
                    gr.components.Radio(
                        SchedulerType._member_names_,
                        label="Schedule",
                        value="V_CRASH",
                    ),
                ]
        with gr.Column():
            audioboxes = []
            gr.Markdown("Output")
            for i in range(MAX_AUDIOBOXES):
                t = gr.components.Audio(label=f"Output #{i+1}", visible=False)
                audioboxes.append(t)

    generate_btn.click(
        fn=variable_outputs,
        inputs=[batch_size_comp, mode_comp, interpolations_comp],
        outputs=audioboxes,
    )
    generate_btn.click(
        fn=generate_audio,
        inputs=[batch_size_comp]
        + [currmodel_comp]
        + [mode_comp]
        + gen_components
        + path_components
        + add_components
        + sampler_components,
        outputs=audioboxes,
    )
