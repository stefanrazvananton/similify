git clone

testat windows 10 gpu e ok

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
