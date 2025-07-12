import torch
import torch.nn as nn

class LSTMModel_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_class = nn.Linear(hidden_size, num_classes)
        self.fc_reg = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn shape: (1, batch, hidden)
        hn = hn.squeeze(0)
        class_out = self.fc_class(hn)
        reg_out = self.fc_reg(hn).squeeze(1)
        return class_out, reg_out


class LSTMModel_1(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layers,num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True)
        self.fc_n = nn.Linear(1,hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc_combined = nn.Linear(hidden_dim*2,hidden_dim)
        self.out_cls = nn.Linear(hidden_dim,num_classes)
    
    def forward(self,x,n):
        lstm_out,_ = self.lstm(x)
        last_out = lstm_out[:,-1,:]
        n_feat = self.fc_n(n)
        combined = torch.cat([last_out,n_feat],dim=1)
        combined = self.dropout(self.fc_combied(combined))
        reg_out = self.out_reg(combined).squeeze(1)
        cls_out = self.out_cls(combined)
        return reg_out, cls_out 


class LSTMModel_2(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layers,num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # n 피처용 MLP
        self.fc_n = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        # 결합 후 MLP
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1)
        )
        self.out_reg = nn.Linear(hidden_dim, 1)
        self.out_cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, n):
        lstm_out, _ = self.lstm(x)
        last_out   = lstm_out[:, -1, :]
        n_feat     = self.fc_n(n)           # now non-linear
        comb       = torch.cat([last_out, n_feat], dim=1)
        comb       = self.fc_combined(comb) # now non-linear + BN
        return self.out_reg(comb).squeeze(1), self.out_cls(comb)

class LSTMModel_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, pred_len):
        super(LSTMModel_3, self).__init__()
        self.pred_len = pred_len  # 예측 시점 수
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 출력 분기
        self.fc_class = nn.Linear(hidden_size, num_classes * pred_len)  # 분류
        self.fc_reg = nn.Linear(hidden_size, pred_len) # 회귀

    def forward(self, x):
        # x: (B, T, F)
        lstm_out, _ = self.lstm(x)  # lstm_out: (B, T, H)
        out = lstm_out[:, -1, :]    # 마지막 시점의 hidden만 사용 (B, H)

        out_class = self.fc_class(out).view(-1, self.pred_len, self.num_classes)  # (B, pred_len, num_classes)
        out_reg = self.fc_reg(out)                                  # (B, pred_len)

        return out_class, out_reg


class BiLSTMModel_0(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True)
        # n 피처 MLP
        self.fc_n = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        # 결합 후 MLP (LSTM 2배 + n 1배 → total: 3배)
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1)
        )
        self.out_reg = nn.Linear(hidden_dim, 1)
        self.out_cls = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, n):
        lstm_out, _ = self.lstm(x)  # (B, T, 2*H)
        last_out = lstm_out[:, -1, :]  # (B, 2*H)
        
        n_feat = self.fc_n(n)  # (B, H)
        
        # Concatenate (B, 3*H)
        combined = torch.cat([last_out, n_feat], dim=1)
        combined = self.fc_combined(combined)
        
        return self.out_reg(combined).squeeze(1), self.out_cls(combined)

