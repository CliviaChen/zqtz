import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# === 可视化中文字体 ===
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# === 配置 ===
SEQ_LEN = 30
EPOCHS = 100
HIDDEN_SIZE = 128  # 增加神经元数量
NUM_LAYERS = 3     # 增加 LSTM 层数
DROPOUT = 0.3
LR = 0.001
SAVE_DIR = "预测结果"
os.makedirs(SAVE_DIR, exist_ok=True)

# === NMSE 计算函数 ===
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    variance = np.var(y_true)
    return mse / variance if variance != 0 else float('inf')

# === LSTM 模型类 ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# === 多股票数据预测函数 ===
def run_lstm_on_stocks(file_list):
    summary = []

    for file in file_list:
        name = os.path.splitext(os.path.basename(file))[0]
        print(f"\n=== 处理股票: {name} ===")

        df = pd.read_excel(file).dropna()

        # 默认特征列和目标列
        feature_cols = ['开盘价(元)', '收盘价(元)', '涨跌幅(%)', '换手率(%)']
        target_cols = ['开盘价(元)', '收盘价(元)']

        # 筛选特征并标准化
        data_all = df[feature_cols].values
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_all)

        # 构造样本（使用所有特征，预测前两个）
        X, y = [], []
        for i in range(len(data_scaled) - SEQ_LEN):
            X.append(data_scaled[i:i + SEQ_LEN])
            y.append(data_scaled[i + SEQ_LEN][:len(target_cols)])
        X, y = np.array(X), np.array(y)

        # 数据划分
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # 转换为张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # 初始化模型（自动适配特征维度）
        input_size = X.shape[2]
        output_size = y.shape[1]
        model = LSTMModel(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                          output_size=output_size, dropout=DROPOUT)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # 训练模型
        for epoch in range(EPOCHS):
            model.train()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 预测 + 评估
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()

        # 还原原始单位
        dummy_extend = data_scaled[:len(y_pred)]
        dummy_extend[:, :output_size] = y_pred
        predicted = scaler.inverse_transform(dummy_extend)[:, :output_size]

        dummy_extend[:, :output_size] = y_test
        actual = scaler.inverse_transform(dummy_extend)[:, :output_size]

        mse = mean_squared_error(actual, predicted)
        nmse = compute_nmse(actual, predicted)

        # 未来30天递推预测
        future_preds = []
        last_seq = torch.tensor(X_test[-1:], dtype=torch.float32)
        with torch.no_grad():
            for _ in range(30):
                pred = model(last_seq)
                pred_full = np.zeros((input_size,))
                pred_full[:output_size] = pred.numpy()
                future_preds.append(pred_full[:output_size])
                pred_tensor = torch.tensor(pred_full).unsqueeze(0).unsqueeze(0).float()
                last_seq = torch.cat((last_seq[:, 1:, :], pred_tensor), dim=1)

        future_preds = np.array(future_preds).reshape(-1, output_size)
        dummy_future = data_scaled[:30].copy()
        dummy_future[:, :output_size] = future_preds
        future_preds_inv = scaler.inverse_transform(dummy_future)[:, :output_size]

        # 保存预测结果
        df_future = pd.DataFrame(future_preds_inv, columns=[f"预测{col}" for col in target_cols])
        df_future.index.name = '未来天数'
        df_future.to_excel(os.path.join(SAVE_DIR, f"{name}_未来30天预测.xlsx"))

        summary.append({
            '股票名称': name,
            '测试集样本数': len(actual),
            'MSE': round(mse, 4),
            'NMSE': round(nmse, 4)
        })

    # 保存汇总表
    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(os.path.join(SAVE_DIR, '多股票预测评估汇总.xlsx'), index=False)
    print("\n所有结果已保存到目录：", SAVE_DIR)

if __name__ == "__main__":
    stock_files = [
        '中兴通讯.xlsx', '中国通号.xlsx', '中科曙光.xlsx', '中芯国际.xlsx',
        '歌尔股份.xlsx', '比亚迪.xlsx', '沪电股份.xlsx', '海信视像.xlsx',
        '海光信息.xlsx', '深南电路.xlsx'
    ]
    run_lstm_on_stocks(stock_files)

