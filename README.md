# LSTM-autoencoder
Deep learning term project

## Introduction
Pytorch implementation of LSTM Autoencoder based Time series anomaly detection. 

<hr>

## Data Preparation
본 프로젝트에서는 두 개의 time series 데이터셋을 사용했습니다.
1. Secure Water Treatment(SWaT) A2 Dataset \[[Request link](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)\]
2. The Almanac of Minutely Power dataset(AMPds2) \[[Download link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FIE0S4)\]
- AMPds2 데이터셋에서는 univariate setting으로 오직 ```WH*.csv``` sensor data만 사용했습니다. 전력에서는 ```P``` column을, 가스와 수도에서는 ```avg_rate``` column을 사용했습니다.
- AMPds2 데이터셋은 anomaly 라벨이 없으므로 본 프로젝트에서는 threshold 값을 기준으로 초과하는 경우에 anomaly로 구분하였습니다. 각 modality의 anomaly수는 앙상블 방법으로 라벨링을 한 [해당 논문](https://ieeexplore.ieee.org/document/8574884)과 일치하도록 threshold를 설정하였습니다.
- 모델 학습을 진행하기 앞서, AMPds2의 경우 라벨링을 먼저 진행하였습니다. 라벨링 코드는 ```AMPds2_labeling.ipynb```을 참고하시기 바랍니다.

<hr>

## Usage
```main.py``` 를 통해 모델을 학습하고 성능을 측정할 수 있습니다. 각 데이터 셋에 대한 사용 예시는 다음과 같습니다.
### SWaT dataset
```
CUDA_VISIBLE_DEVICES=0 python main.py --save_path <your save path> --data_path <path to dataset> \
                                      --seq_length 60 --trn_shift_length 1 --tst_shift_length 60 \
                                      --hidden_dim 128 --num_layers 2 --attention False --epoch 100 
```
- ```data_path``` : ```normal_v1.csv```, ```attack_v0.csv``` 파일이 저장된 디렉토리의 경로를 인자로 받습니다.

### AMPds2 dataset(Electricity 예시)
```
CUDA_VISIBLE_DEVICES=0 python main.py --save_path <your save path> --data_path <path to dataset> \
                                      --seq_length 10 --trn_shift_length 1 --tst_shift_length 10 \
                                      --hidden_dim 64 --num_layers 2 --attention False --epoch 20 
```
- ```data_path``` : 라벨 컬럼이 추가된 ```Electricity_WHE.csv``` 파일의 경로를 인자로 받습니다.
- ```attention``` : Encoder의 마지막 LSTM layer의 hidden state에 self attention을 적용한 후 decoder의 입력에 반영합니다.
- ```save_path``` : 결과 파일과 마지막 에폭에서의 모델 state를 저장할 경로를 인자로 받습니다. 저장 결과는 아래와 같습니다.

```
<your save path> / train.log        # epoch, train loss, iteration time, data time
                 / val.log          # epoch, validation loss
                 / loss_curve.png   # Train & validation loss curve.
                 / last.pth         # model state at the last epoch.
                 / recon_err.npy    # reconstruction error(L1) on test set. Index means time step.
                 / score_list.npy   # anomaly score on test set. Index means time step.
                 / label_list.npy   # label of test set. Index means time step.
                 / perf_dict.json   # auroc, precision, recall, f1-score, mean reconstruction error(L1)
```

<hr>

## Arguments
|Argument|Default|Description|
|--------|-------|-----------|
|save_path|'./result'|Save path|
|data_path|-|Path to dataset|
|seq_length|60|Sequence length|
|trn_shift_length|1|Shift length to generate Train data|
|tst_shift_length|60|Shift length to generate Validation & Test data. Should be same as seq_length|
|hidden_dim|128|Latent dimension|
|num_layers|2|Number of LSTM layers|
|batch_size|1024|Batch size|
|epoch|100|Train epoch|
|lr|0.001|Learning rate|
|attention|False|Whether to apply attention|

<hr>

## Results

- 각 데이터 셋의 성능 평가 결과입니다. Precision, Recall, F1-score는 static threshold을 기준으로 계산됩니다. 
- Static threshold는 test dataset에 대한 anomaly score의 평균 + 3\* 표준편차로 설정됩니다.(3-sigma rule)

|Dataset|attention|AUROC|Precision|Recall|F1-score|
|-------|---------|-----|---------|------|--------|
|SWaT|False|0.8170|0.9973|0.5849|0.7374|
|SWaT|True|0.8130|0.9978|0.5850|0.7376|
|AMPds2-Electricity|False|0.9889|0.9399|0.2451|0.3889|
|AMPds2-Electricity|True|0.9968|0.9617|0.2673|0.4182|
|AMPds2-Gas|False| | | | |
|AMPds2-Gas|True| | | | |
|AMPds2-Water|False| | | | |
|AMPds2-Water|True| | | | |

## Reference
[(LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection, 2016, Malhotra et al)](https://arxiv.org/abs/1607.00148)

