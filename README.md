[![Python 3.7.9](https://img.shields.io/badge/python-3.7.9-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe)
[![Tensorflow 1.15](https://img.shields.io/badge/tensorflow-1.15-orange.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://pypi.org/project/tensorflow/1.15.0/)
[![Ambev-tech](https://img.shields.io/badge/Ambev--tech-2021-yellow.svg?style=for-the-badge)](https://ambevtech.gupy.io/)
![OWL](https://img.shields.io/badge/OWL-1.0-brown?style=for-the-badge&logo=OWL&logoColor=white)


# Transfer learning com Tensorflow API para detecção de objetos

Esse repositório apresenta um tutorial completo sobre como utilizar um modelo tensorflow já treinado para classificar objetos personalizados no google colab. Aqui é descrito como preparar as imagens para que a partir delas conseguiremos criar os arquivos `.xml`, `.tfRecord` que são fundamentais para o re-treino do nosso modelo.

**Inicie clonando esse repository**: 

```
  $ git clone https://github.com/jeffreire/transfer-learning-TF.git
```

**importe os pré-requisitos do projeto:**
```
$ pip install -r requirements.txt
```

## step-to-step

### 1 - Selecionar as imagens
Antes de mais nada, selecionamos as imagens dos objetos que usaremos para treinar e testar o modelo, ao produzir todas as imagens devemos armazenalas na pasta imagens localizada em ```.\models\research\object_detection\images```. Agora, com as imagens produzidas e armazenadas no lucar correto, devemos definir um tamanho padrão para cada imagem. Portanto, para isso usaremos o script ```transform_image_resolution.py``` localizado em ```.\models\research\object_detection```. Dentro da pasta, `object_detection/`, execute: (altere `width` e `heigth` na linha de comando pelo tamanho preferido, exemplo: `800 600`).
```
python transform_image_resolution.py -d imagens/ -s width height
```
enfim, depois de definimos o tamnho padrão das images devemos dividir as imagens de treinamento e imagens de teste, para isso, criamos as pastas `imagens/train` e `imagens/test` com suas respectivas imagens.

### 2 -  Rotulando os objetos nas imagens
Agora, depois de prepararmos as imagens, o proximo passo é rotularmos os objetos em cada uma das imagens e, assim, criarmos os asquivos `pascalvoc` para cada uma delas. Para isso utilizaremos a interface `labelImg`.
  
Para obeter o script, acesse: 

[![GITHUB - LABELIMG](https://img.shields.io/badge/labelImg-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tzutalin/labelImg)

Depois de rotular os objetos, na pasta train quanto a pasta test deverão estar estruturada na seguinte forma:
```
  /imagens/train/                                                    
                 image_1.jpg                              
                 image_1.xml
                 image_2.jpg
                 image_2.xml
                 .
                 .
                 .
                 image_n.jpg
                 image_n.xml      

  /imagens/train/                                                   
                 image_1.jpg                              
                 image_1.xml
                 image_2.jpg
                 image_2.xml
                 .
                 .
                 .
                 image_n.jpg
                 image_n.xml
```    


### 3 - Gerando os arquivos CSV, RECORD E PBTXT.

Em seguinda, depois de criarmos o nosso primeiro arquivo do tipo, `pascalvoc` o próximo passo é converter-lo para arquivo do tipo `CSV`, para isso usaremos o script `xml_to_csv.py` que está localizado em `models/research/object_detection/`, na pasta `object_detection` execute o seguinte comando: 
```
$ python xml_to_csv.py
```
Ao executar, automaticamente será convertido todos os arquivos `.xml` de train e teste para `csv` e serão armazenados em research/object-detection/imagens.

com os arquivos train_labels.csv e test_labels.csv gerados, proximo passo é gerar a conversão dos arquivos do tipo `.CSV` para `RECORD`, para isso, amtes de você executar a linha de comando, precisará realizar uma alteração dentro do sript. Abra o script em um editor e encontre a função, abaixo: 
```
def class_text_to_int(row_label):
    if row_label == 'classe_1':
        return 1
    elif row_label == 'classe_2':
        return 2
    else:
        None;

```
Ótimo, nessa parte do código você deverá adicionar `elif()` de acordo com a quantidade de classes que você pretende prever, para cada classe você retorna um int, como no exemplo, acima. Agora, executaremos o seguinte comando: dentro da pasta `object_detection/` para converter os arquivos de train, execute:
```
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
```
**para converter os arquivos de teste, execute:**
```
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```
Por fim, criamos de forma manual o ultimo arquivo, chamado de `label_map.pbtxt`, esse arquivo dará um mapeamento ao modelo, para isso crie essa estrutura dentro do arquivo:
de acordo com a quantidade de classes que seu modelo preverá e de acordo com os retornos que você alterou dentro na função `class_text_to_int()`.
```
    item {
      id: 1
      name: 'classe_1'
    }

    item {
      id: 2
      name: 'classe_2'
    }
    
```
Caso seu modelo seja multiclasses, esse arquivo terá as mesmas quantidade de classes.
Portanto, finalizamos a parte de estruturação de arquivos, agora devemos armazenar os arquivos nos lugares corretos. Portanto, segue onde cada arquivo deverá ser armazenado:

Os arquivos `test.record` e `train.record` deverão estar dentro da pasta `object_detection`:
```
 models/research/object_detection
```
Já o arquivo `label_map.pbtxt` será armazenado dentro da pasta `training/`:
```
 models/research/object_detection/training
```

### 4 - Obter o Modelo

Nesse exemplo estamos utilizando o modelob faster_rcnn_inception_v2_coco_2018_01_28, então, para obter-lo, acesse: 
[![faster__rcnn__inception__v2__coco-tensorflow--1.15](https://img.shields.io/badge/faster__rcnn__inception__v2__coco-tensorflow--1.15-blue?style=for-the-badge&logo=tensorflow&logoColor=white)](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)

Ao baixar, extráia o arquivo no diretório `research\object_detection\` e assim etará pronto.

### 5 - Alterando o arquivo faster_rcnn_inception_v2_pets.config
O arquivo que estou usando como exemplo é o `faster_rcnn_inception_v2_pets.config` mas indepedente do arquivo que você usa, fique atento a essas mudanças:

**Primeira alteração** - Abra o arquivo em um editor de texto, entre as linhas 7 e 14, você verá o seguinte código:
```
model {
  faster_rcnn {
    num_classes: 2
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
      }
```
Na variável `num_classes: ` altere a quantidade de classes de acordo com suas classes.

**Segunda alteração** - No mesmo arquivo na linha 106, você verá o seguinte código:
```
 fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt": "
```
Nessa parte do código você colocará como fine_tune_checkpoint:  o caminho do diretório do modelo, ou seja:
```
research\object_detection\faster_rcnn_inception_v2_coco_2018_01_28
```
**Terceira alteração** - Alteração do diretório do nosso arquivo `record`. Nas linhas de código 121 a 126 e 133 a 137, você verá: 

linha 121-126:
```
train_input_reader: {
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/pet_faces_train.record-?????-of-00010"
  }
  label_map_path: "PATH_TO_BE_CONFIGURED/pet_label_map.pbtxt"
}
```
linha 133-137
```
eval_input_reader: {
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/pet_faces_val.record-?????-of-00010"
  }
  label_map_path: "PATH_TO_BE_CONFIGURED/pet_label_map.pbtxt"
```
- [x] \(Na linha 123 a variavél `input_path: `receberá o diretório do arquivo `train.record`.)
- [x] \(Na linha 125 a variavél `label_map_path: ` receberá o diretório do arquivo `label_map.pbtxt"`.)
- [x] \(Na linha 135 a variavél `input_path: ` receberá o diretório do arquivo `test.record`.)
- [x] \(Na linha 137 a variavél `label_map_path: ` receberá o diretório do arquivo `label_map.pbtxt"`.)

# Iniciando o Treinamento do Modelo

Enfim, depois de pra´pararmos todo o nossas imagens, arquivos e modelos, agorea é hora de testar o nosso modelo e ve se realmente vai desenvolver um bom desempenho no treinamento. 

Para treinar o modelo, dentro da pasta `object_detection/` abra o terminal e execute:
```
$ train.py --logtostdeer --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

Se tudo ocorreu bem, o treinamneto do seu modelo foi um sucesso.

# Exportando inference graph

Depois de treinarmos o nosso modelo, devemos entao exportar o a inference graph, e para isso, execute o seguinte comando: 
```
python export_inference_graph.py --input_type image_tensor /
                                 --pipeline_config_path training/faster_rcnn_inception_v2_pets.config /
                                 --trained_checkpoint_prefix training/model.ckpt-xxxx /
                                 --output_directory inference_graph
```
**Atenção** que no comando `model.ckpt-xxxx` onde está os `xxxx` irá o número do arquivo que seu modelo irá gerar, e para descobrir, vá atá a pasta `training/` e obtenha o arquivo `model.ckpt-` de maior numero e anote no lugar do `XXXX`.

Ao executar esse comando, surgirá dentro da sua pasta `models/research/object_detection/` uma pasta chamada de `inference_graph` nela constará todos os arquivos gerados do treinamento.
