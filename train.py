import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMModel, Trainer


def normalize_features(train_df, val_df, feature_cols):
    # 删除常数列
    valid_cols = [col for col in feature_cols if train_df[col].nunique() > 1]
    removed = set(feature_cols) - set(valid_cols)
    if removed:
        print(f"⚠️ 以下特征被移除（无效或常数）：{list(removed)}")

    feature_cols = valid_cols
    train_df.dropna(subset=feature_cols + ['close'], inplace=True)
    val_df.dropna(subset=feature_cols + ['close'], inplace=True)

    scaler = MinMaxScaler()
    full_cols = feature_cols + ['close']
    train_df[full_cols] = scaler.fit_transform(train_df[full_cols])
    val_df[full_cols] = scaler.transform(val_df[full_cols])

    return train_df, val_df, scaler, feature_cols


def create_sequences(df, feature_cols, target_col, window_size, predict_days=5):
    features = df[feature_cols].values
    targets = df[target_col].values

    X, y = [], []
    for i in range(len(features) - window_size - predict_days + 1):
        X.append(features[i:i+window_size])  # 使用过去20天的所有特征数据
        y.append(targets[i+window_size:i+window_size+predict_days])  # 预测未来5天的收盘价

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def preprocess_and_create_dataloaders(train_df, val_df, feature_cols, target_col='close', window_size=20, batch_size=32, predict_days=5):
    train_df, val_df, scaler, cleaned_cols = normalize_features(train_df, val_df, feature_cols)

    train_dataset = create_sequences(train_df, cleaned_cols, target_col, window_size, predict_days)
    val_dataset = create_sequences(val_df, cleaned_cols, target_col, window_size, predict_days)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler, cleaned_cols


def load_data(train_path, val_path):
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("数据文件未找到，请检查路径。")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df


def main():
    train_path = "./training_data/train_data.csv"
    val_path = "./training_data/val_data.csv"
    model_save_path = "./saved_models"
    os.makedirs(model_save_path, exist_ok=True)

    feature_columns = ["open", "high", "low", "volume", "amount", "turn", "pctChg"]
    window_size = 20
    epochs = 100
    learning_rate = 0.001  
    batch_size = 512
    predict_days = 5 

    print("📥 加载数据...")
    train_df, val_df = load_data(train_path, val_path)

    print("🧹 数据预处理...")
    train_loader, val_loader, _, cleaned_cols = preprocess_and_create_dataloaders(
        train_df, val_df, feature_columns, window_size=window_size, batch_size=batch_size, predict_days=predict_days
    )

    print(f"✅ 有效特征列：{cleaned_cols}")
    print("📊 训练样本数:", len(train_loader.dataset))
    print("📊 验证样本数:", len(val_loader.dataset))

    print("🧠 初始化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    model = LSTMModel(input_size=len(cleaned_cols), hidden_size=64, num_layers=2, output_size=predict_days).to(device)

    print("🛠️ 初始化训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        device=device,
        save_path=os.path.join(model_save_path, "best_model.pth"),
        patience=10  # 加大 early stopping 容忍度
    )

    print("🏋️ 开始训练...")
    trained_model = trainer.train()

    final_model_path = os.path.join(model_save_path, "model_final.pth")
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"✅ 模型训练完成，已保存至 {final_model_path}")


if __name__ == "__main__":
    main()
