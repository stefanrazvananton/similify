Testat pe Windows 10 si Ubuntu 24.04.1 GPU/CPU
GPU testate: GTX 1660, GTX 1060, RTX 3080, RTX 3080 Ti

Daca se ruleaza pe GPU acesta trebuie sa aiba minim 2 GB VRAM
! De implementat selectia automata CPU daca GPU nu are minim 2


descarca arhiva "data" si dezarhiveaza in folder-ul curent
[https://drive.google.com/file/d/1OIcse4cG9pZ2u-do5PZtzabffw4J_gop/view?usp=sharing](https://drive.google.com/drive/folders/1G0QzBbvXwsseRG2pP00NVewOMW88wLib?usp=sharing)


Pentru a rula
git clone ...
cd similify

conda create -n "similify" python=3.10.15
conda activate similify


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
https://pytorch.org/get-started/locally/




pip install flask
pip install flask_socketio
pip install scikit-learn
pip install tqdm
pip install matplotlib

python app.py

127.0.0.1:5000
