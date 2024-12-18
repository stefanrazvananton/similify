Testat pe Windows 10 si Ubuntu 24.04.1 atat pe GPU cat si pe CPU



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
Descarcati dataset-ul procesat de pe [Google Drive](https://drive.google.com/drive/folders/1JF-A-LgbF8mH5EjMorGdWi1iIlKflDOs?usp=sharing) si mutati in folder-ul './similify'.

Structura trebuie sa fie '.similify/imgs/0', '.similify/imgs/1', '.similify/imgs/2' ... 


### Modelele preantrenate:
Descarcati modelele preantrenate contrastiv de pe [Google Drive](https://drive.google.com/drive/folders/198Dfq5g0ZZbsjDYi6yu_KHf0VxQYGsoi?usp=sharing), si mutati-le in folder-ul './similify'

### Caracteristici (features) precalculate:
Descarcati caracteristicile de pe [Google Drive](https://drive.google.com/drive/folders/11hBGEKtb2-oSvrJOF77EdgcGq4zUcSx6?usp=sharing), si mutati-le in folder-ul './similify'

Daca se vrea, caracteristicile se pot genera din nou prin executarea comenzilor
```
python generate_features_normal_models.py
python generate_features_contrastive_models.py
```


### Testare:
Executati comanda ```python app.py```

Dupa un timp se vor afisa in consola doua adrese
```
http://127.0.0.1:5000
xxx.xxx.x.xxx:5000
```
Accesati una dintre aceste adrese pentru a utiliza aplicatia

