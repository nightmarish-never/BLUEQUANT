import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model import LSTMModel


def preprocess_data(test_data, feature_columns, target_col='close', scaler=None):
    test_data.dropna(subset=feature_columns + [target_col], inplace=True)
    valid_cols = [col for col in feature_columns if test_data[col].nunique() > 1]
    feature_columns = valid_cols

    if scaler is None:
        scaler = MinMaxScaler()
        test_data[feature_columns + [target_col]] = scaler.fit_transform(test_data[feature_columns + [target_col]])
    else:
        test_data[feature_columns + [target_col]] = scaler.transform(test_data[feature_columns + [target_col]])

    return test_data, scaler, feature_columns


def create_test_sequences(data, feature_columns, window_size, predict_days=5):
    features = data[feature_columns].values
    X = []
    for i in range(len(features) - window_size - predict_days + 1):
        X.append(features[i:i + window_size])  # 用过去window_size天的数据
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    return X_tensor


def backtest(model, test_tensor, device):
    model.eval()
    test_tensor = test_tensor.to(device)

    with torch.no_grad():
        predictions = model(test_tensor).cpu().numpy()  # (N, 5)
    return predictions


def inverse_scale_close(predicted, scaler, feature_columns):
    """
    predicted: shape (N, predict_days)，为归一化后的close预测
    返回反归一化后的 close 值: 同 shape
    """
    predicted = np.atleast_2d(predicted)
    num_features = len(feature_columns)
    dummy = np.zeros((predicted.shape[0], num_features + 1))  # 包含 close 共 n+1 列

    results = []
    for i in range(predicted.shape[1]):  # 每一列预测（对应未来第 i 天）
        dummy_copy = dummy.copy()
        dummy_copy[:, -1] = predicted[:, i]  # 把 close 放最后一列
        inversed = scaler.inverse_transform(dummy_copy)
        results.append(inversed[:, -1])  # 提取反归一化后的 close 值

    return np.stack(results, axis=1)  # shape: (N, predict_days)


def evaluate_predictions(actual, predicted):
    for i in range(predicted.shape[1]):
        y_true = actual[:, i]
        y_pred = predicted[:, i]
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        print(f"📉 第{i+1}天预测 MSE: {mse:.6f}, RMSE: {rmse:.6f}")


def main():
    window_size = 20
    predict_days = 5
    target_col = 'close'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = LSTMModel(input_size=7, hidden_size=64, num_layers=2, output_size=predict_days).to(device)
    model.load_state_dict(torch.load('./saved_models/model_final.pth', map_location=device))

    # 加载测试数据
    test_data = pd.read_csv("./training_data/test_data.csv")
    feature_columns = ["open", "high", "low", "volume", "amount", "turn", "pctChg"]

    # 数据归一化
    test_data, scaler, feature_columns = preprocess_data(test_data, feature_columns, target_col)

    # 创建测试序列
    test_tensor = create_test_sequences(test_data, feature_columns, window_size, predict_days)
    print(f"🧪 测试样本数量: {len(test_tensor)}")

    # 模型预测（归一化值）
    predicted_norm = backtest(model, test_tensor, device)

    # 反归一化预测结果
    predicted = inverse_scale_close(predicted_norm, scaler, feature_columns)

    # 构造实际未来5天的close作为评估目标
    actual = []
    close_values = test_data[target_col].values
    for i in range(len(test_tensor)):
        actual.append(close_values[i + window_size : i + window_size + predict_days])
    actual = np.array(actual)

    # 反归一化实际 close 值
    actual = inverse_scale_close(actual, scaler, feature_columns)

    # 评估
    evaluate_predictions(actual, predicted)


if __name__ == "__main__":
    main()
