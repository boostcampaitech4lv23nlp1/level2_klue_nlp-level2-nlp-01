# level2_klue_nlp-level2-nlp-01

## stratified k-fold cross validation
- 분류 문제
- 데이터 불균형으로인한 데이터 편향이 나타남

👉 기본 k-fold 교차검증으로는 성능평가가 잘 되지 않을 수 있음

따라서 **label 별 분포도에 따라** 데이터를 나누는 stratified k-fold 교차검증 적용


### kfold 실험 결과

- micro f1 : 72.8062   ->   74.1241
- auprc : 77.8146   ->   80.5488


**kfold 적용 후 약 2% 성능향상을 확인할 수 있었음**
