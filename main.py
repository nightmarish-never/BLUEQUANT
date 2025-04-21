import os
import torch
import baostock as bs
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel

# 全局常量
FEATURE_COLUMNS = ["open", "high", "low", "volume", "amount", "turn", "pctChg"]
WINDOW_SIZE = 20
PREDICT_DAYS = 5
MODEL_PATH = "./saved_models/model_final.pth"

# 登录 baostock
def login_baostock():
    lg = bs.login()
    print(f'✅ 登录状态: {lg.error_code} - {lg.error_msg}')
    return lg

def logout_baostock():
    bs.logout()
    print("👋 已登出 baostock")

# 拉取股票历史数据
def fetch_stock_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="1"
    )

    if rs.error_code != '0':
        raise Exception(f"[错误] 查询失败：{rs.error_msg}")

    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)

    float_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
    df[float_cols] = df[float_cols].astype(float)
    return df

# 单支股票预测
def predict(model_path: str, data: pd.DataFrame, feature_columns: list, window_size: int, predict_days: int) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 清洗数据
    data = data.dropna(subset=feature_columns + ['close']).copy()
    if data.shape[0] < window_size:
        raise ValueError(f"数据不足 {window_size} 天，无法预测。")

    # 保存 close 的最值用于反归一化
    close_min = data['close'].min()
    close_max = data['close'].max()

    # 归一化（不包括 close）
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[feature_columns])
    data_scaled_df = pd.DataFrame(data_scaled, columns=feature_columns)

    # 构建最近的序列
    recent_sequence = data_scaled_df[feature_columns].values[-window_size:]
    recent_tensor = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    # 初始化模型并加载权重
    model = LSTMModel(input_size=len(feature_columns), hidden_size=64, num_layers=2, output_size=predict_days)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        prediction = model(recent_tensor).cpu().numpy().reshape(-1)

    print("预测的归一化值：", prediction)

    # 用 close 的 min/max 反归一化
    predicted_prices = prediction * (close_max - close_min) + close_min
    print("反归一化后的值：", predicted_prices.tolist())

    return predicted_prices.tolist()


# 主流程
def main():
    os.makedirs("data", exist_ok=True)
    login_baostock()

    end_date = date.today()
    start_date = end_date - timedelta(days=3 * 365)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    while True:
        code = input("\n📥 请输入股票代码（如: sh.600000，输入 q 退出）：").strip().lower()
        if code in ['q', 'quit', 'exit']:
            break

        try:
            df = fetch_stock_data(code, start_date_str, end_date_str)
            if df.empty:
                print(f"[提示] 未获取到 {code} 的数据。")
                continue

            # 保存数据
            file_name = code.replace('.', '_') + ".csv"
            path = os.path.join("data", file_name)
            df.to_csv(path, index=False)
            print(f"✅ 历史数据已保存至 {path}")

            if len(df) < WINDOW_SIZE + PREDICT_DAYS:
                print("[提示] 数据不足以进行预测（至少需要25天）。")
                continue

            # 预测
            predict(model_path=MODEL_PATH, data=df, feature_columns=FEATURE_COLUMNS,
                    window_size=WINDOW_SIZE, predict_days=PREDICT_DAYS)

        except Exception as e:
            print(f"[异常] {e}")

    logout_baostock()


if __name__ == "__main__":
    main()
