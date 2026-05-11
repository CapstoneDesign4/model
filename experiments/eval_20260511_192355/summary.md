# 평가 요약 — 2026-05-11T19:24:22

- data_dir: `C:\CapstoneDesign\model\data\sample`
- 총 파일 수: 18
- 위험(positive): 12, 비위험(negative): 6
- 임계값 sweep: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

## 라벨 분포
- baby_cry: 3
- glass_shatter: 3
- negative: 6
- siren: 3
- vehicle_horn: 3

## 임계값별 지표

| threshold | TP | FP | FN | TN | precision | recall | F1 | FAR |
|-----------|----|----|----|----|-----------|--------|----|-----|
| 0.10 | 12 | 1 | 0 | 5 | 0.9231 | 1.0000 | 0.9600 | 0.1667 |
| 0.20 | 11 | 1 | 1 | 5 | 0.9167 | 0.9167 | 0.9167 | 0.1667 |
| 0.30 | 11 | 1 | 1 | 5 | 0.9167 | 0.9167 | 0.9167 | 0.1667 |
| 0.40 | 11 | 1 | 1 | 5 | 0.9167 | 0.9167 | 0.9167 | 0.1667 |
| 0.50 | 11 | 1 | 1 | 5 | 0.9167 | 0.9167 | 0.9167 | 0.1667 |
| 0.60 | 10 | 1 | 2 | 5 | 0.9091 | 0.8333 | 0.8696 | 0.1667 |
| 0.70 | 9 | 1 | 3 | 5 | 0.9000 | 0.7500 | 0.8182 | 0.1667 |
| 0.80 | 9 | 1 | 3 | 5 | 0.9000 | 0.7500 | 0.8182 | 0.1667 |
| 0.90 | 7 | 1 | 5 | 5 | 0.8750 | 0.5833 | 0.7000 | 0.1667 |

## 라벨 그룹별 클래스 score (mean / max)

### label = `baby_cry` (n=3)

| class | mean | max |
|-------|------|-----|
| screaming | 0.1185 | 0.2179 |
| baby_cry | 0.6639 | 0.9867 |
| glass_shatter | 0.0013 | 0.0029 |
| breaking | 0.0000 | 0.0000 |
| gunshot | 0.0002 | 0.0004 |
| explosion | 0.0052 | 0.0118 |
| fire_alarm | 0.0004 | 0.0008 |
| smoke_alarm | 0.0000 | 0.0001 |
| siren | 0.0007 | 0.0019 |
| civil_defense_siren | 0.0001 | 0.0003 |
| car_alarm | 0.0000 | 0.0000 |
| vehicle_horn | 0.0011 | 0.0026 |

### label = `glass_shatter` (n=3)

| class | mean | max |
|-------|------|-----|
| screaming | 0.0087 | 0.0205 |
| baby_cry | 0.0026 | 0.0064 |
| glass_shatter | 0.8113 | 0.9472 |
| breaking | 0.7836 | 0.8871 |
| gunshot | 0.0269 | 0.0702 |
| explosion | 0.0748 | 0.1926 |
| fire_alarm | 0.0027 | 0.0064 |
| smoke_alarm | 0.0017 | 0.0048 |
| siren | 0.0034 | 0.0068 |
| civil_defense_siren | 0.0006 | 0.0010 |
| car_alarm | 0.0034 | 0.0097 |
| vehicle_horn | 0.0132 | 0.0328 |

### label = `negative` (n=6)

| class | mean | max |
|-------|------|-----|
| screaming | 0.0041 | 0.0109 |
| baby_cry | 0.0024 | 0.0065 |
| glass_shatter | 0.0043 | 0.0186 |
| breaking | 0.0021 | 0.0086 |
| gunshot | 0.0131 | 0.0619 |
| explosion | 0.1808 | 0.9717 |
| fire_alarm | 0.0002 | 0.0004 |
| smoke_alarm | 0.0001 | 0.0001 |
| siren | 0.0024 | 0.0047 |
| civil_defense_siren | 0.0004 | 0.0007 |
| car_alarm | 0.0003 | 0.0004 |
| vehicle_horn | 0.0188 | 0.0514 |

### label = `siren` (n=3)

| class | mean | max |
|-------|------|-----|
| screaming | 0.0087 | 0.0249 |
| baby_cry | 0.0027 | 0.0069 |
| glass_shatter | 0.0000 | 0.0001 |
| breaking | 0.0000 | 0.0000 |
| gunshot | 0.0000 | 0.0000 |
| explosion | 0.0038 | 0.0107 |
| fire_alarm | 0.0021 | 0.0057 |
| smoke_alarm | 0.0003 | 0.0006 |
| siren | 0.8610 | 0.9814 |
| civil_defense_siren | 0.1612 | 0.2462 |
| car_alarm | 0.0066 | 0.0106 |
| vehicle_horn | 0.1502 | 0.4381 |

### label = `vehicle_horn` (n=3)

| class | mean | max |
|-------|------|-----|
| screaming | 0.0027 | 0.0075 |
| baby_cry | 0.0008 | 0.0023 |
| glass_shatter | 0.0003 | 0.0008 |
| breaking | 0.0000 | 0.0001 |
| gunshot | 0.0002 | 0.0007 |
| explosion | 0.0018 | 0.0053 |
| fire_alarm | 0.0001 | 0.0003 |
| smoke_alarm | 0.0000 | 0.0000 |
| siren | 0.2187 | 0.6542 |
| civil_defense_siren | 0.0002 | 0.0005 |
| car_alarm | 0.0895 | 0.2216 |
| vehicle_horn | 0.9350 | 0.9982 |

산출 파일:
- predictions.jsonl
- metrics.csv