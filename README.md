# Transfer learning com Tensorflow API para detecção de objetos

Esse repositório apresenta um tutorial completo sobre como utilizar um modelo tensorflow já treinado para classificar objetos personalizados no google colab. Aqui é descrito como preparar as imagens para que a partir delas conseguiremos criar os arquivos `.xml`, .tfRecord que sao essencias para o re-treino do nosso modelo.

## Passos

### 1 - Selecionar as imagens
Antes de tudo selecionamos as imagens dos objetos que desejamos treinar o modelo, e assim deixamos o tamanho da imgens de forma padronizada, dessa forma facilitamos o treinamento do modelo.

### 2 -  Rotulando os objetos nas imagens
O segundo passo depois de prepararmos as imagens é rotularmos os objetos e assim criarmos os arquivos `.xml`, ou melhor dizendo, os arquivos `pascalvoc`. Temos duas opções para criarmos esses arquivos.
* Opção 1 - A primeira opção é utilizar o script `opencv_object_tracking.py` para rastrearmos os objetos atraves de imagens ou videos e partir dele criarmos os arquivos `pascalvoc`, para isso utilizamos a linha de comando, onde você pode utilizar o comando `--video` se voce deseja rastrar objetos e criar samples a partir de videos. Caso deseja rastrear a partir de imagens, você poderá utilizar o camando `--imagesdirectory`.
```
python opencv_object_tracking.py --video da --tracker csrt --vocxml True
```
**Ao executr o comando de video para fazer a marcação da caixa delimitadora no objeto, recisa precionar a tecla `W`**

* Opção 2 - A segunda opção pode ser a melhor quando se referimos em detectar objetos em imagens e não em videos. Nessa segunda opção, utilzariamos o **LabelImg**.  LabelImg é uma ótima ferramenta para rotular imagens e sua página GitHub tem instruções muito claras sobre como instalar e usar.

LabelImg GitHub link](https://github.com/tzutalin/labelImg)

[LabelImg download link](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1)

Os nomes dos arquivos `.xml` necessitam ter o mesmo nome das imagens que referenciam. Por exemplo: Suponhamos que você deseja treinar um modelo que detecta 2 classes:
* **classe 1: cachorro** 
* **classe 2: gato**

Assim, os arquivos serão organizados dessa forma:
* **cachorro_1.png, cachorro_1.xml, ..., cachorro_N.jpg, cachorro_N.xml.** 
* **gato_1.png, gato_1.xml, ..., gato_n.jpg, gato_n.xml.**

e assim por diante.

### 3 -  Organizando em pastas
Depois de completar o passo 2 em todas as imagens criadas, armazenaremos as imagens e os arquivos .xml em uma unica pasta. ou seja, criaremos uma pasta chamada `data`. Depois, criaremos treis subpastas.
* **data/imagens** - Onde armazenará as pastas train e test.
- **data/imagens/train e data/imagens/test** - De forma intercalada ambas as pastar terão os dados armazenados na seginte forma: 
```
  cachorro_1.jpg
  cachorro_1.xml
  cachorro_2.jpg
  cachorro_2.xml
  ...
  cachorro_n.jpg
  cachorro_n.xml
  ...
  gato_1.jpg
  gato_1.xml
  gato_2.jpg
  gato_2.xml
  ...
  gato_n.jpg
  gato_n.xml
```
Ma mesma pasta `data` teremos uma sub pasta nomeada de `data/annotations` onde a principio estará vazia, mas posteriormente receberá os arquivos `train_labels.csv, teste_labels.csv, train.record, teste.record`. Mas já chegaremos lá. Nesse passo apenas criaremos a pasta.

E por fim, criaremos uma terceira subpasta na pasta `data` noeada de `data/val` onde armazenaremos imagens para nossa validação final.

### 4 - Gerando os arquivos CSV, RECORD E PBTXT.

Depois de rotularmos os objetos e gerarmos os arquivos ´pascalvoc´, precisamos gerar os três ultimos arquivos. Para isso usaremos os scripts  `xml_to_csv.py` e `generate_tfrecord.py` para gerarmos o `CSV` e o `RECORD`, respectivamente.
Para gerarmos o arquivo de `test_labels.csv`, execute essa linha de comando, utilizando o script `xml_to_csv.py`:
```
 python xml_to_csv.py -i ./data/images/test -o ./data/annotations/test_labels.csv
```
Para gerar o train_labels.csv, execute essa linha de comando:
```
 python xml_to_csv.py -i ./data/images/train -o ./data/annotations/train_labels.csv
```
Depois de gerarmos os arquivos `csv`, proximo passo sera criar os arquivos `record`. Para isso, vamos utilizar o script  `generate_tfrecord.py`, execute o comando a seguir para gerar o arquivo `test.record`.
```
generate_tfrecord.py --csv_input=./data/images/test_labels.csv  --image_dir=./data/images/test --output_path=test.record
```
Para gerar os arquivos de `train.record` execute a linha de comando a seguir:
```
generate_tfrecord.py --csv_input=./data/images/train_labels.csv  --image_dir=./data/images/train --output_path=train.record
```
Por fim, criamos de forma manual o ultimo arquivo, chamado de `label_map.pbtxt`, esse arquivo dara um mapeamento ao modelo, onde com o nosso exemplo binario entre cachorro e gato, poderemos dizer para o nosso modelo. Caso retorne 1, então é cachorro. Caso retorne 2, então gato. portanto, segue o modelo que sera ser criado:
```
    item {
      id: 1
      name: 'cachorro'
    }

    item {
      id: 2
      name: 'gato'
    }
    
```
Caso seu modelo seja multiclasses, esse arquivo terá as mesmas quantidade de classes.
Portanto, finalizamos a parte de estruturação de arquivos, agora devemos armazenar os arquivos em lugares corretos. Entao, os arquivos `CSVs, record e pbtxt` serão armazenados dentro da pasta `./data/annotations`.

### 5 - 
