import torch
import numpy as np

# 예측 펑션
def predict_future(station_name, evse_name, df_raw, col_transformer, model, device='cuda'):
    model.eval()
    target_reg = 'kwh_per_usage_time'
    # 1. 입력 데이터 필터링 (최근 24개만 사용)
    evse_list = df_raw['evse_name'].to_list()
    station_location_list = df_raw['station_location'].to_list()
    
    # station-evse 조합 존재 확인
    match = df_raw[
        (df_raw['station_location'] == station_name) &
        (df_raw['evse_name'] == evse_name)
    ]
    
    if station_name not in df_raw['station_location'].unique():
        raise ValueError(f"❌ '{station_name}'이라는 충전소는 존재하지 않습니다.")
    if evse_name not in df_raw['evse_name'].unique():
        raise ValueError(f"❌ '{evse_name}'이라는 기기는 존재하지 않습니다.")
    
    match = df_raw[(df_raw['station_location'] == station_name) & (df_raw['evse_name'] == evse_name)]
    if match.empty:
        raise ValueError(f"❌ '{station_name}'의 '{evse_name}' 기기에 대한 데이터가 존재하지 않습니다.")
    
    df_target = match
    df_recent = df_target.sort_values('charging_start_time_ts').iloc[-24:]  # 시간 기준 정렬 +마지막 24시간 시퀀스
    
    if len(df_recent) < 24:
        raise ValueError("해당 기기(station-evse)의 데이터가 24개보다 적습니다.")
    
    X_input_raw = df_recent.copy()  

    # 3. 전처리 적용
    X_input_trans = col_transformer.transform(X_input_raw)  # (24, feature_dim)
    
    # 4. 배치 형태로 변환
    X_input_tensor = torch.tensor(X_input_trans, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 24, feature_dim)

    # 5. 예측 수행
    with torch.no_grad():
        pred_cls, pred_reg = model(X_input_tensor)  # pred_cls: (1, 24, num_classes), pred_reg: (1, 24)

    # 6. 결과 정리
    pred_cls_label = pred_cls.argmax(dim=2).squeeze(0).cpu().numpy()  # (24,)
    pred_reg_val = pred_reg.squeeze(0).cpu().numpy()  # (24,)

    # 회귀 결과 역변환
    for name, transformer, cols in col_transformer.transformers_:
        if isinstance(cols, list) and target_reg in cols:
            # 해당 컬럼의 인덱스를 찾아서 슬라이싱해 inverse
            col_idx = cols.index(target_reg)
            if hasattr(transformer, 'inverse_transform'):
                pred_reg_val_full = np.zeros((pred_reg_val.shape[0], len(cols)))
                pred_reg_val_full[:, col_idx] = pred_reg_val
                pred_reg_val_inv_full = transformer.inverse_transform(pred_reg_val_full)
                pred_reg_val = pred_reg_val_inv_full[:, col_idx]
                break
    else:
        raise ValueError("회귀 스케일러를 ColumnTransformer에서 찾을 수 없습니다.")
    
    # 정확히 n시간뒤 1개 값만 받고 싶을경우
    n =5
    cls_n = pred_cls_label[n-1]
    reg_n = pred_reg_val[n-1]

    return pred_cls_label, pred_reg_val