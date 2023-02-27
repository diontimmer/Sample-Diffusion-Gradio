import gradio as gr
import os, sys
import torch, torchaudio
import gc

sys.path.append('sample_diffusion')
from util.util import load_audio, cropper
from util.platform import get_torch_device_type
from dance_diffusion.api import RequestHandler, Request, RequestType, SamplerType, SchedulerType, ModelType



# ****************************************************************************
# *                                 Settings                                 *
# ****************************************************************************

max_audioboxes = 100
modelfolder = 'models'



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
    return [f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f)) and f.endswith('.ckpt')]


def make_audio_outputs(amount):
    audio_outputs = []
    for i in range(amount):
        audio_outputs.append(gr.components.Audio(label=f'Batch #{i+1}'))
    return audio_outputs

def variable_outputs(k):
    k = int(k)
    return [gr.Audio.update(visible=True)]*k + [gr.Audio.update(visible=False)]*(max_audioboxes-k)

# ****************************************************************************
# *                                 Generate                                 *
# ****************************************************************************

def generate_audio(batch_size, model, mode,use_autocast, use_autocrop, device_accelerator, device_offload,  sample_rate, chunk_size, seed, tame,audio_source, audio_target, mask, noise_level, interpolations_linear, interpolations, resamples, keep_start, steps, sampler, schedule, progress=gr.Progress(track_tqdm=True)
    ):
    # casting
    gc.collect()
    torch.cuda.empty_cache()
    request_type = RequestType[mode]
    model_type = ModelType.DD
    sampler_type = SamplerType[sampler]
    scheduler_type = SchedulerType[schedule]

    # load model
    modelpath = f'{modelfolder}/{model}'

    device_type_accelerator = device_accelerator if(device_accelerator != None) else get_torch_device_type()
    device_accelerator = torch.device(device_type_accelerator)
    device_offload = torch.device(device_offload)

    request_handler = RequestHandler(device_accelerator, device_offload, optimize_memory_use=False, use_autocast=use_autocast)
    seed = int(seed) if(seed!=-1) else torch.randint(0, 4294967294, [1], device=device_type_accelerator).item()
    autocrop = cropper(chunk_size, True) if(use_autocrop==True) else lambda audio: audio

    # make request
    request = Request(
        request_type=request_type,
        model_path=modelpath,
        model_type=model_type,
        model_chunk_size=int(chunk_size),
        model_sample_rate=int(sample_rate),
        
        seed=int(seed),
        batch_size=int(batch_size),
        
        audio_source=autocrop(load_audio(device_accelerator,audio_source, sample_rate)) if(audio_source != None) else None,
        audio_target=autocrop(load_audio(device_accelerator,audio_target, sample_rate)) if(audio_target != None) else None,
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

    # save audio
    outputs = save_audio((0.5 * response.result).clamp(-1,1) if(tame == True) else response.result, f"Output/{ModelType.DD.__str__()}/{mode.__str__()}/", sample_rate, f"{seed}")
    outputs += ['data/dummy.mp3'] * (max_audioboxes - len(outputs))
    return outputs



# ****************************************************************************
# *                                   MAIN                                   *
# ****************************************************************************


def main():
    with gr.Blocks() as dd_ui:
        with gr.Row():
            with gr.Column():
                gr.Markdown("Sample Generator")
                generate_btn = gr.Button(label="Generate")
                with gr.Tab('General Settings'):
                    models = load_models()
                    currmodel_comp = gr.components.Dropdown(models, label="Model Checkpoint", value=models[0])
                    batch_size_comp = gr.components.Slider(label="Batch Size", value=1, maximum=max_audioboxes, minimum=1, step=1)
                    gen_components = [
                        gr.components.Radio(RequestType._member_names_, label="Mode of Operation", value="Generation"),
                        gr.components.Checkbox(label="Use Autocast", value=True),
                        gr.components.Checkbox(label="Use Autocrop", value=True),
                        gr.components.Radio(["cpu", "cuda"], label="Device Accelerator", value="cuda"),
                        gr.components.Radio(["cpu", "cuda"], label="Device Offload", value="cuda"),
                        gr.components.Number(label="Sample Rate", value=48000),
                        gr.components.Number(label="Chunk Size", value=65536),
                        gr.components.Number(label="Seed", value=-1),
                        gr.components.Checkbox(label="Tame", value=True)
                        ]
                with gr.Tab('Variation/Interpolation Settings'):
                    with gr.Row():
                        path_components = [
                            gr.File(label="Audio Source Path", interactive=True, file_count="single", file_types=[".mp3", ".wav", ".flac"], elem_id="audio_source_path_file"),
                            gr.File(label="Audio Target Path", interactive=True, file_count="single", file_types=[".mp3", ".wav", ".flac"], elem_id="audio_target_path_file"),
                            gr.File(label="Audio Mask Path", interactive=True, file_count="single", file_types=[".mp3", ".wav", ".flac"], elem_id="audio_mask_path_file"),
                        ]
                    add_components = [
                        #variation/interp settings
                        gr.components.Slider(label="Noise Level", value=0.7, maximum=1, minimum=0),
                        gr.components.Number(label="Interpolations Linear", value=None),
                        gr.components.Textbox(label="Interpolation Positions (comma-separated)", value=None),
                        gr.components.Slider(label="Resampling Steps", value=4, maximum=50, minimum=1),
                        gr.components.Checkbox(label="Keep Start of Audio", value=True),
                        ]
                with gr.Tab('Sampler Settings'):  
                        sampler_components = [  
                        #extra settings
                        gr.components.Slider(label="Steps", value=50, maximum=100, minimum=10),
                        gr.components.Radio(SamplerType._member_names_, label="Sampler", value="IPLMS"),
                        gr.components.Radio(SchedulerType._member_names_, label="Schedule", value="CrashSchedule"),
                        ]
            with gr.Column():
                audioboxes = []
                gr.Markdown("Output")
                for i in range(max_audioboxes):
                        t = gr.components.Audio(label=f"Batch #{i+1}", visible=False)
                        audioboxes.append(t)



        generate_btn.click(fn=variable_outputs, inputs=batch_size_comp, outputs=audioboxes)
        generate_btn.click(fn=generate_audio, inputs=[batch_size_comp] + [currmodel_comp] + gen_components + path_components + add_components + sampler_components, outputs=audioboxes)
        dd_ui.queue()
        dd_ui.launch()


if __name__ == "__main__":
    main()