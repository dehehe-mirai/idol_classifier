# Million Live Idol Cl@ssifier!

`IDOLM@STER Million Live!` 아이돌 사진 분류기

### 주의사항

* 이미지는 전부 복사됩니다. 충분한 저장 공간이 있는지 확인하세요.

* CPU 많이 먹습니다.

* GIF 아직 지원 안함

### 다운로드

[Download](https://github.com/dehehe-mirai/idol_classifier/releases/download/1.0/dist_cpu.zip)
또는 화면 우측의 `Release`에서 확인한다.

### 사용 방법

* 분류할 이미지들이 있는 폴더를 선택한다.

* 분류할 이미지들이 복사될 폴더를 선택한다.

* 기다린다.

### 알려진 문제점

* 디렉터리 경로에 유니코드 문자가 포함된 경우 에러가 발생한다

* `Akane` 라벨의 인덱스가 0번이라서 오탐률이 높다

### 디렉터리 구조

```
core
 \- * # 얼굴 감지기 관련
configs
 \- * # 얼굴 감지기 관련
models
 \- * # 얼굴 감지기 관련
label
 \- label.txt # 이미지 라벨
model
 \- model.tflite # TFLite 모델
raw
 \- 분류되어있는 원본 사진
cropped
 \- 전처리된 분류 사진
idol_models
 \- Keras 모델

detect.py # LFFD 얼굴 감지기 테스트용

bulk_convert.py # 이미지 전처리용
train.py # 이미지 훈련 및 TFlite 모델 생성

label_image.py # TFLite 분류기
predict.py # 해당 파일
```

### 훈련 방법

* `raw`, `cropped` 폴더를 생성한 뒤 `raw` 폴더 아래에 `/label/labels.txt`에 맞춘 이름의 아이돌 이름 폴더를 생성한다.

* 이미지를 분류해 `raw` 폴더 아래의 아이돌 이름 폴더에 넣는다.

* 이미지를 전처리한다.
```
python bulk_convert.py
```
* `TFLite` 모델을 생성한다.

```
python train.py
```

* 동작이 잘 되는지 검증한다.

```
python predict.py
```

### 빌드 방법

`pyinstaller`를 이용해 다음 명령어를 입력한다.

```
pyinstaller -F --add-data "./Lib/site-packages/mxnet/*.dll;./mxnet" .\predict.py
```

이후 생성된 `exe` 파일을 포함한 폴더에 얼굴 감지기 관련 폴더, `model` 폴더를 복사한 뒤  배포한다.

### GPU 빌드 방법

얼굴 분류 시 CUDA 가속을 사용하려면, 

```
pip install mxnet-cu90
```
을 입력하고,

`core/detector/detector.py`의 다음 줄을
```
import mxnet as mx
```

다음과 같이 수정한다.

```
import mxnet-cu90 as mx
```

### 참조한 Repository 및 문서

[](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub?hl=ko)
[](https://github.com/freedomofkeima/transfer-learning-anime)
[](https://github.com/cheese-roll/light-anime-face-detector)
