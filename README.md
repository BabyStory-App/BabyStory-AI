# BabyStory AI

[BabyStory App](https://github.com/BabyStory-App)의 AI 레포지토리입니다.

![poster3](https://raw.githubusercontent.com/BabyStory-App/.github/refs/heads/main/assets/poster/poster.pdf3.jpg)

![poster5](https://raw.githubusercontent.com/BabyStory-App/.github/refs/heads/main/assets/poster/poster.pdf5.jpg)

울음을 감지하기 위한 YamNet 모델을 활용하는 코드와 울음 원인을 분석하기 위하여 데이터를 전처리하고 ResNet50을 Fine-tuning하여 분류를 수행하는 코드가 포함되어 있습니다.

전체 화면 스크린샷은 [BabyStory App Screenshots](https://github.com/BabyStory-App/.github/tree/main/assets/Screenshots)에서, 시연 영상은 [BabyStory App Demo](https://github.com/BabyStory-App/.github/tree/main/assets/Screen%20recordings)에서 확인 가능합니다.

### 데이터 수집 및 전처리

데이터는 "donate a cry dataset", "위쟈오구(wojiaoguodekai) 아기 울음 인식 dataset" 외 5개의 데이터셋을 수집, 제공 받았습니다. 데이터를 수집하고 전처리하는 과정은 [get_data](https://github.com/BabyStory-App/BabyStory-AI/tree/main/get_data)의 README를 참고해주시기 바랍니다.

음성 데이터를 전처리하는 과정은 [음성 데이터 전처리.ipynb](https://github.com/BabyStory-App/BabyStory-AI/blob/main/%E1%84%8B%E1%85%B3%E1%86%B7%E1%84%89%E1%85%A5%E1%86%BC%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%8E%E1%85%A5%E1%84%85%E1%85%B5.ipynb)에서 확인 가능합니다.

### 울음 감지 및 분석

- 울음 감지: On-device에서 울음을 감지하기 위해 TensorFlow Lite를 사용하였으며, MobileNet 아키텍처 기반의 YamNet 모델(521가지 소리 분류)을 활용합니다.
- 울음 분석: 음성 파일로부터 추출한 MFCC 데이터를 기반으로, Fine-tuning된 ResNet50 모델을 통해 7가지 울음 원인을 분류합니다.
  - 울음은 아래와 같은 7가지 원인으로 분류됩니다.
    - Awake: 막 깨어남
    - Diaper: 기저귀 교체가 필요함
    - Hungry: 배고픔
    - Sleepy: 졸림
    - Hug: 안아달라고 함
    - Sad: 슬픔
    - Uncomfortable: 불편함

울음 감지 프로세스는 아래와 같습니다.

![cry_detect_process1](https://raw.githubusercontent.com/BabyStory-App/.github/refs/heads/main/assets/cry_detect_process1.png)

해당 과정의 순서도는 다음과 같습니다.

![cry_detect_process2](https://raw.githubusercontent.com/BabyStory-App/.github/refs/heads/main/assets/cry_detect_process2.png)
