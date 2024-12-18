Testat pe Windows 10 si Ubuntu 24.04.1 atat pe GPU cat si pe CPU

Pentru a intelege modul de functionare accesati [prezentarea cu explicatii minimale](https://github.com/stefanrazvananton/similify/blob/main/explicatii%20minimale.pdf))

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

Clasele de la 0 la 9 sunt cele din setul de test de la [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

Clasa 10 este formata din setul de date [manga109](http://www.manga109.org/en/).

Clasele 11 si 12 sunt volumele 1, respectiv 2 din [Usogui](https://archive.org/details/manga-0v3r-usogui-v01-49-complete/Usogui).

Clasele 13 si 14 sunt primele 1000 de imagini din setul de train pentru competitia [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/).

### Modelele preantrenate:
Descarcati modelele preantrenate contrastiv de pe [Google Drive](https://drive.google.com/drive/folders/198Dfq5g0ZZbsjDYi6yu_KHf0VxQYGsoi?usp=sharing), si mutati-le in folder-ul './similify'

### Caracteristici (features) precalculate:
Descarcati caracteristicile de pe [Google Drive](https://drive.google.com/drive/folders/11hBGEKtb2-oSvrJOF77EdgcGq4zUcSx6?usp=sharing), si mutati-le in folder-ul './similify'

Daca se vrea, caracteristicile se pot genera din nou prin executarea comenzilor (nu este recomandat daca nu aveti la dispozitie o placa video cu minim 6 GB VRAM)
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

