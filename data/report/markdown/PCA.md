# 2차 PCA
## Robustscaler 사용
- 중앙값 기준 정규화 (이상치 민감도 ↓)
- 이상치 영향 최소화, 주성분에서 가장 많은 분산 보존
### PCA 결과

- 선택된 주성분 수: 3
- 각 주성분 보존율: [0.67055733 0.11416112 0.03745587]
- 총 보존율: 0.822174326584484

>▶️ PC1 (설명력:0.6706) 상위 기여 변수<br/>
  scaling__expected_departure_time_ts → 기여도: 0.9805<br/>
  scaling__charging_end_time_ts  → 기여도: 0.0770<br/>
  scaling__charging_start_time_ts → 기여도: 0.0770<br/>
  scaling__expected_departure_time_missing → 기여도: -0.0571<br/>
  scaling__usage_departure_time_diff_missing → 기여도: -0.0571<br/>

>▶️ PC2 (설명력:0.1142) 상위 기여 변수<br/>
  scaling__charging_end_time_ts  → 기여도: 0.6883<br/>
  scaling__charging_start_time_ts → 기여도: 0.6883<br/>
  scaling__expected_departure_time_ts → 기여도: -0.1082<br/>
  scaling__requested_kwh         → 기여도: 0.0762<br/>
  scaling__delivered_kwh         → 기여도: 0.0729<br/>

>▶️ PC3 (설명력:0.0375) 상위 기여 변수<br/>
  scaling__usage_departure_time_diff_ts → 기여도: 0.6852<br/>
  scaling__post_charge_departure_delay_ts → 기여도: 0.5958<br/>
  scaling__expected_time_diff_ts → 기여도: -0.2266<br/>
  onehot__post_charge_departure_range_1 → 기여도: -0.1958<br/>
  scaling__actual_usage_duration_ts → 기여도: 0.1490<br/>

### PCA 조정 
- 선택된 주성분 수: 7
- 각 주성분 보존율: [0.67055733 0.11416112 0.03745587 0.02734959 0.02331214 0.015148380.01357226]
- 총 보존율: 0.9015566968875924

>▶️ PC1 (설명력:0.6706) 상위 기여 변수<br/>
  scaling__expected_departure_time_ts → 기여도: 0.9805<br/>
  scaling__charging_end_time_ts  → 기여도: 0.0770<br/>
  scaling__charging_start_time_ts → 기여도: 0.0770<br/>
  scaling__expected_departure_time_missing → 기여도: -0.0571<br/>
  scaling__usage_departure_time_diff_missing → 기여도: -0.0571<br/>

>▶️ PC2 (설명력:0.1142) 상위 기여 변수<br/>
  scaling__charging_end_time_ts  → 기여도: 0.6883<br/>
  scaling__charging_start_time_ts → 기여도: 0.6883<br/>
  scaling__expected_departure_time_ts → 기여도: -0.1082<br/>
  scaling__requested_kwh         → 기여도: 0.0762<br/>
  scaling__delivered_kwh         → 기여도: 0.0729<br/>

>▶️ PC3 (설명력:0.0375) 상위 기여 변수<br/>
  scaling__usage_departure_time_diff_ts → 기여도: 0.6852<br/>
  scaling__post_charge_departure_delay_ts → 기여도: 0.5958<br/>
  scaling__expected_time_diff_ts → 기여도: -0.2266<br/>
  onehot__post_charge_departure_range_1 → 기여도: -0.1958<br/>
  scaling__actual_usage_duration_ts → 기여도: 0.1490<br/>
