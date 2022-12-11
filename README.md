# **Naver Boostcamp AI Tech level 2 NLP 1조**
## **[NLP] 문장 내 게체간 관계 추출**

관계 추출은 문장의 단어에 대한 속성과 관계를 예측하여

지식 그래프 구축에 있어 핵심적으로 역할을 하는 단계입니다.

더불어 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은

다양한 자연어 처리 응용 연구 분야에서 중요하게 역할할 수 있습니다.

<br><br><br>

## **Contributors**

|이름|id|역할|
|:--:|:--:|--|
|이용우|[@wooy0ng](https://github.com/wooy0ng)|**PM** (협업 리딩, 역할 지원)|
|강혜빈|[@hyeb](https://github.com/hyeb)|**코더** (Model Architecture 설계 및 개선)
|권현정|[@malinmalin2](https://github.com/malinmalin2)|**데이터 분석** (데이터 전처리 및 증강)|
|백인진|[@eenzeenee](https://github.com/eenzeenee)|**코더** (Model Architecture 설계 및 개선)|
|이준원|[@jun9603](https://github.com/jun9603)|**코더** (Model Architecture 설계 및 개선)


<br><br><br>


## **Data**

기본 데이터
- train set : 25,976개
- validation set : 3,247개
- evaluation set : 3,247개

✓ 평가 데이터의 50%는 public 점수 계산에 반영되어 실시간 리더보드에 표기된다.

✓ 나머지 50%는 private 점수 계산에 반영되어 대회 종료 후 평가된다. 


<br><br><br>


## **Experiments**

|idx|experiment|  
|:--:|--|
|1|Data Preprocessing (중복 행 제거, 반복되는 괄호 데이터 제거)|
|2|Easy Data Augmentatiion|
|3|AEDA|
|4|back translation|
|5|bert mean pooling|
|6|focal loss|
|7|contrastive loss|
|8|entity marker (typed entity marker, typed entity marker punctuation)
|9| R-BERT
|10| RECENT
|11| ensemble



<br><br><br>

## **project tree**

```
template
├── dataloader.py
├── dict_label_to_num.pkl
├── dict_num_to_label.pkl
├── inference.py
├── losses.py
├── metrics.py
├── models.py
├── requirements.txt
└── train.py
```

<br><br><br>

## **Train**

```
$ python main.py --augment [value]
```

### **augment**
- `--tokenizer_name` : huggingface tokenizer name (str)
- `--model_name` : huggingface model name (str)
- `--batch_size` : batch_size (int)
- `--max_epoch` : epoch_size (int)
- `--learning_rate` : learning rate (float)
- `--marker` : entity masking (bool)
- `--shuffle` : shuffle train dataset (bool)
- `--criterion` : select loss function [cross_entropy, focal_loss]
- `--n_folds` :  number of folds (int)
- `--split_seed` : fold dataset's split seed (int)
- `--train_path` : train dataset's path (str)
- `--dev_path` : validation dataset's path (str)
- `--test_path` : evaluation dataset's path (str)
- `--predict_path` : prediction dataset's path (str)

<br><br><br>

## **Inference**

```
$ python inference.py --augment [value]
```

### **augment**
- `--tokenizer_name` : huggingface tokenizer name (str)
- `--model_name` : huggingface model name (str)
- `--predict_path` : prediction dataset's path (str)


<br><br><br>


