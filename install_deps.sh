# run following after conda activate

source ~/.bashrc
pip uninstall bitsandbytes -y
pip install --upgrade bitsandbytes
pip install deepspeed
pip install h5py
pip install flash-attn --no-build-isolation
pip uninstall deepspeed -y
pip install deepspeed==0.15.4
pip install gdown
pip install -e .

git config --global user.email "nganngants@gmail.com"
git config --global user.name "nganngants"

gdown --folder https://drive.google.com/drive/folders/1-8EYXtqvcj9wet80WNuAjYW-E9p5--7Q?usp=drive_link

git submodule update --init
cd VideoLLaMA2
git checkout audio_visual
pip install -e .
cd ..
# gdown 17nHhoVx8im-fjsY7cqFL2OrMwVusHMHJ

# unzip mamba_ckpt.zip