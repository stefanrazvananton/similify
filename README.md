Testat pe Windows 10 si Ubuntu 24.04.1



Descarca toate fisierele de [aici](https://drive.google.com/drive/folders/1G0QzBbvXwsseRG2pP00NVewOMW88wLib?usp=sharing)


Pentru a rula
git clone https://github.com/stefanrazvananton/similify.git
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
