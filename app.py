import gradio as gr
import os, sys
import torch, torchaudio
import gc

sys.path.append('sample_diffusion')
from util.util import load_audio, crop_audio
from util.platform import get_torch_device_type
from dance_diffusion.api import RequestHandler, Request, RequestType, SamplerType, SchedulerType, ModelType



# ****************************************************************************
# *                                 Settings                                 *
# ****************************************************************************

max_audioboxes = 100
modelfolder = 'models' if ( os.getenv('SDGFOLDER') is None ) else os.getenv('SDGFOLDER')

# ****************************************************************************
# *                                  Helpers                                 *
# ****************************************************************************



def save_audio(audio_out, output_path: str, sample_rate, id_str:str = None):
    files=[]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(output_path, f"sample_{id_str}_{ix + 1}.wav" if(id_str!=None) else f"sample_{ix + 1}.wav")
        open(output_file, "a").close()
        output = sample.cpu()

        torchaudio.save(output_file, output, int(sample_rate))
        files.append(output_file)
    return files


def load_models():
    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(modelfolder):
        for filename in filenames:
            if filename.endswith('.ckpt'): 
              list_of_files.append( os.sep.join([dirpath.replace(modelfolder,''), filename])[1:])
    return list_of_files


def refresh_all_models(*inputs):
        models = load_models()
        selected = models[0]
        return gr.Dropdown.update(value=selected, choices=models)

def make_audio_outputs(amount):
    audio_outputs = []
    for i in range(amount):
        audio_outputs.append(gr.components.Audio(label=f'Batch #{i+1}'))
    return audio_outputs

def variable_outputs(output_amt, mode, interp_amount):
    output_amt = int(output_amt) if (mode != 'Interpolation') else int(output_amt) * int(interp_amount)

    return [gr.Audio.update(visible=True)]*output_amt + [gr.Audio.update(visible=False)]*(max_audioboxes-output_amt)

refresh_symbol = '\U0001f504'

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

class ToolButtonTop(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, with extra margin at top, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool-top", **kwargs)

    def get_block_name(self):
        return "button"

class FormRow(gr.Row, gr.components.FormComponent):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "row"

# ****************************************************************************
# *                                 Generate                                 *
# ****************************************************************************

def generate_audio(batch_size, model, mode,use_autocast, crop_offset, device_accelerator, device_offload,  sample_rate, chunk_size, seed, tame,audio_source, audio_target, mask, noise_level, interpolations_linear, interpolations, resamples, keep_start, steps, sampler, schedule, progress=gr.Progress(track_tqdm=True)
    ):
    # casting
    gc.collect()
    torch.cuda.empty_cache()
    request_type = RequestType[mode]
    model_type = ModelType.DD
    sampler_type = SamplerType[sampler]
    scheduler_type = SchedulerType[schedule]
    audio_source = audio_source.name if(audio_source != None) else None
    audio_target = audio_target.name if(audio_target != None) else None
    mask = mask.name if(mask != None) else None
    crop_offset = int(crop_offset)

    # load model
    modelpath = f'{modelfolder}/{model}'

    device_type_accelerator = device_accelerator if(device_accelerator != None) else get_torch_device_type()
    device_accelerator = torch.device(device_type_accelerator)
    device_offload = torch.device(device_offload)

    request_handler = RequestHandler(device_accelerator, device_offload, optimize_memory_use=False, use_autocast=use_autocast)
    seed = int(seed) if(seed!=-1) else torch.randint(0, 4294967294, [1], device=device_type_accelerator).item()
    
    crop = lambda audio: crop_audio(audio, chunk_size, crop_offset) if crop_offset is not None else audio
    load_input = lambda source: crop(load_audio(device_accelerator, source, sample_rate)) if source is not None else None

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
        mask=torch.load(mask) if(mask != None) else None,
        
        noise_level=noise_level,
        interpolation_positions=interpolations if(interpolations_linear == None) else torch.linspace(0, 1, int(interpolations_linear), device=device_accelerator),
        resamples=int(resamples),
        keep_start=keep_start,
                
        steps=int(steps),
        
        sampler_type=sampler_type,
        sampler_args={'use_tqdm': True},
        
        scheduler_type=scheduler_type,
        scheduler_args={}
    )

    # process request
    response = request_handler.process_request(request)
    outputs = save_audio((0.5 * response.result).clamp(-1,1) if(tame == True) else response.result, f"Output/{ModelType.DD.__str__()}/{mode.__str__()}/", sample_rate, f"{seed}")
    outputs += ['data/dummy.mp3'] * (max_audioboxes - len(outputs))
    return outputs



# ****************************************************************************
# *                                   MAIN                                   *
# ****************************************************************************


def main():
    with gr.Blocks(title='Sample Diffusion') as dd_ui:
        with gr.Row():
            with gr.Column():
                gr.Markdown("Sample Diffusion")
                models = load_models()
                with gr.Column(variant='panel'):
                    currmodel_comp = gr.components.Dropdown(models, label="Model Checkpoint", value=models[0])
                    # refresh_models = gr.Button(value=refresh_symbol, variant='tool')
                    # refresh_models.style(full_width=False)
                    # refresh_models.click(refresh_all_models, currmodel_comp, currmodel_comp)
                    mode_comp = gr.components.Radio(RequestType._member_names_, label="Mode of Operation", value="Generation")
                    generate_btn = gr.Button(value='Generate Samples', label="Generate", variant='primary')
                with gr.Tab('General Settings'):
                    batch_size_comp = gr.components.Slider(label="Batch Size", value=1, maximum=max_audioboxes, minimum=1, step=1)
                    gen_components = [
                        gr.components.Checkbox(label="Use Autocast", value=True),
                        gr.components.Number(label="Crop Offset", value=0),
                        gr.components.Radio(["cpu", "cuda"], label="Device Accelerator", value="cuda"),
                        gr.components.Radio(["cpu", "cuda"], label="Device Offload", value="cuda"),
                        gr.components.Number(label="Sample Rate", value=48000),
                        gr.components.Number(label="Chunk Size", value=65536),
                        gr.components.Number(label="Seed", value=-1),
                        gr.components.Checkbox(label="Tame", value=True)
                        ]
                with gr.Tab('Variation/Interpolation Settings'):
                    with gr.Row(variant='panel'):
                        path_components = [
                            gr.File(label="Audio Source Path", interactive=True, file_count="single", file_types=[".mp3", ".wav", ".flac"], elem_id="audio_source_path_file"),
                            gr.File(label="Audio Target Path", interactive=True, file_count="single", file_types=[".mp3", ".wav", ".flac"], elem_id="audio_target_path_file"),
                            gr.File(label="Audio Mask Path", interactive=True, file_count="single", file_types=[".mp3", ".wav", ".flac"], elem_id="audio_mask_path_file"),
                        ]
                    noise_level_comp = gr.components.Slider(label="Noise Level", value=0.7, maximum=1, minimum=0)
                    interpolations_comp = gr.components.Number(label="Interpolations Linear", value=3)
                    interpolations_pos_comp =  gr.components.Textbox(label="Interpolation Positions (comma-separated)", value=None)
                    resamples_comp = gr.components.Slider(label="Resampling Steps", value=4, maximum=50, minimum=1)
                    keep_start_comp = gr.components.Checkbox(label="Keep Start of Audio", value=True)
                    add_components = [noise_level_comp, interpolations_comp, interpolations_pos_comp, resamples_comp, keep_start_comp]
                with gr.Tab('Sampler Settings'):  
                        sampler_components = [  
                        #extra settings
                        gr.components.Slider(label="Steps", value=50, maximum=250, minimum=10),
                        gr.components.Radio(SamplerType._member_names_, label="Sampler", value="IPLMS"),
                        gr.components.Radio(SchedulerType._member_names_, label="Schedule", value="CrashSchedule"),
                        ]
            with gr.Column():
                audioboxes = []
                gr.Markdown("Output")
                for i in range(max_audioboxes):
                        t = gr.components.Audio(label=f"Output #{i+1}", visible=False)
                        audioboxes.append(t)



        generate_btn.click(fn=variable_outputs, inputs=[batch_size_comp, mode_comp, interpolations_comp], outputs=audioboxes)
        generate_btn.click(fn=generate_audio, inputs=[batch_size_comp] + [currmodel_comp] + [mode_comp] + gen_components + path_components + add_components + sampler_components, outputs=audioboxes)
        dd_ui.queue()
        dd_ui.launch(share=True)


if __name__ == "__main__":
    if os.path.exists(modelfolder) and len(os.listdir(modelfolder)) > 0:
        main()
    else:
        input('ERROR: Please place a model in the models folder or change the models path in the script file!')
        exit()
