Testat pe Windows 10 si Ubuntu 24.04.1



## Utilizare:
### Environment recomandat:
```
Python 3.10.15
Pytorch 2.5.1
torchvision 0.20.1
```

### Pasi pentru instalare:
```
git clone https://github.com/stefanrazvananton/similify.git
cd similify
conda create -n "similify" python=3.10.15
conda activate similify
```

Instalati Pytorch folosind instructiuniile de [aici](https://pytorch.org/get-started/locally/).

Folositi ```pip install -r requirements.txt``` pentru a instala dependentele.


### Dataset:
Descarcati dataset-ul procesat de pe [Google Drive](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) si mutati in folder-ul './similify'.

Structura trebuie sa fie '.similify/imgs/0', '.similify/imgs/1', '.similify/imgs/2' ... 


### Modelele preantrenate:
Descarcati modelele preantrenate contrastiv de pe [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing), si mutati-le in folder-ul './similify'

### Caracteristici (features) precalculate:
Descarcati caracteristicile de pe [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing), si mutati-le in folder-ul './similify'



### Testare:
Executati comanda ```python app.py```


pip install flask

pip install flask_socketio

pip install scikit-learn

pip install tqdm

pip install matplotlib





python app.py

dupa se intra pe server-ul afisat pe ecran

127.0.0.1:5000
