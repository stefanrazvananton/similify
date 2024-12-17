Testat pe Windows 10 si Ubuntu 24.04.1






Pentru a rula

git clone https://github.com/stefanrazvananton/similify.git

cd similify

conda create -n "similify" python=3.10.15

conda activate similify

Daca am GPU

SE POATE MODIFICA IN FUNCTIE DE VERSIUNEA CUDA

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Daca nu am CPU

pip install torch torchvision torchaudio





pip install flask

pip install flask_socketio

pip install scikit-learn

pip install tqdm

pip install matplotlib



1) Descarca toate fisierele de [aici](https://drive.google.com/drive/folders/1G0QzBbvXwsseRG2pP00NVewOMW88wLib?usp=sharing)
   
3) Dezarhiveaza si pune toate fisierele in folder-ul similify
   
5) Dezarhiveaza  "imgs.zip"
   
7) Folder-ul imgs trebuie sa aiba structura

imgs:
-0

-1

-2

-3

-4

...

python app.py

dupa se intra pe server-ul afisat pe ecran

127.0.0.1:5000
