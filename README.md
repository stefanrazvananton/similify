Testat pe Windows 10 si Ubuntu 24.04.1 GPU/CPU
GPU testate: GTX 1660, GTX 1060, RTX 3080, RTX 3080 Ti

Daca se ruleaza pe GPU acesta trebuie sa aiba minim 2 GB VRAM
! De implementat selectia automata CPU daca GPU nu are minim 2



Pentru a rula
git clone



cd similify

descarca arhiva "data" si dezarhiveaza in folder-ul curent
https://drive.google.com/file/d/1OIcse4cG9pZ2u-do5PZtzabffw4J_gop/view?usp=sharing

conda create -n "similify" python=3.10.15
conda activate similify


daca am gpu pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
daca nu am gpu pip install torch torchvision torchaudio - de testat


pip install flask
pip install flask_socketio
pip install scikit-learn
pip install tqdm -< poate il scot e nevoie doar atunci cand extrag features pe tot setul de date, in rest nu am nevoie de loading bar
pip install matplotlib -< nevoie doar cand am facut debug

pythn app.py

127.0.0.1:5000
