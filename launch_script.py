# this scripts installs necessary requirements and launches main program
import subprocess
import os
import importlib.util
import sys

# ****************************************************************************
# *                                   UTIL                                   *
# ****************************************************************************

python = sys.executable if os.getenv('SDGPYTHON') is None else os.getenv('SDGPYTHON')
platform = sys.platform

def prRed(skk): print(f"\033[91m{skk}\033[00m") 
def prGreen(skk): print(f"\033[92m{skk}\033[00m")
def prYellow(skk): print(f"\033[93m{skk}\033[00m")

def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")

def is_installed(package):
    if package in aliases:
        package = aliases[package]
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run_pip(args, desc=None):
    index_url = os.environ.get('INDEX_URL', "")
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")



# ****************************************************************************
# *                                  CONFIG                                  *
# ****************************************************************************

SKIP_INSTALL = False

torch_command = "pip install torch==1.13.1+cu117 torchaudio --extra-index-url https://download.pytorch.org/whl/cu117" if platform != 'win32' else 'pip install torch torchaudio'
main_script_path = "app"
pre_torch_packages = []
post_torch_packages = [
                        'gradio',
                        'PySoundFile', 
                        'black', 
                        'diffusers',  
                        'tqdm',
                        ]

aliases = {
    "PySoundFile": 'soundfile'
}

prGreen('Starting launch script...')

if __name__ == "__main__":

    # INSTALL

    if not SKIP_INSTALL:

        # pre torch packages
        if pre_torch_packages:
            for package in pre_torch_packages:
                if not is_installed(package):
                    run_pip(f"install {package}", package)


        # TORCH INSTALL
        if not is_installed("torch") and torch_command is not None:
            run(f'"{python}" -m {torch_command}', "Installing torch.", "Couldn't install torch", live=True)
        
        # post torch packages
        if post_torch_packages:
            for package in post_torch_packages:
                if not is_installed(package):
                    run_pip(f"install {package}", package)
                    
        if not os.path.exists('sample_diffusion'):
            run(f'git clone https://github.com/sudosilico/sample-diffusion sample_diffusion', "Cloning sample-diffusion repo.", "Couldn't clone sample-diffusion repo", live=True)

    # LAUNCH
    run(f'"{python}" -m {main_script_path}', "Starting main script..", "Couldn't start main script!", live=True)
