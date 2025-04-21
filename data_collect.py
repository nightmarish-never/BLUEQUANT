import baostock as bs
import pandas as pd
import numpy as np
import os
from datetime import date
from sklearn.model_selection import train_test_split

def login_baostock():
    lg = bs.login()
    print(f'login respond error_code: {lg.error_code}')
    print(f'login respond error_msg: {lg.error_msg}')
    return lg

def fetch_hs300_stocks():
    rs = bs.query_hs300_stocks()
    hs300_stocks = []
    while rs.error_code == '0' and rs.next():
        hs300_stocks.append(rs.get_row_data())
    return pd.DataFrame(hs300_stocks, columns=rs.fields)

def download_stock_data(stock_codes, data_dir):
    stock_data = []
    for stock_code in stock_codes:
        try:
            rs = bs.query_history_k_data_plus(stock_code, "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST", start_date='', end_date=date.today().strftime("%Y-%m-%d"), frequency="d", adjustflag="1")
            data_list = []
            while rs.error_code == '0' and rs.next():
                data_list.append(rs.get_row_data())
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df['stock_code'] = stock_code  # 添加股票代码列
                stock_data.append(df)
                print(f"✅ 下载 {stock_code} 数据成功")
        except Exception as e:
            print(f"❌ 下载 {stock_code} 数据失败: {e}")
    
    return stock_data

def preprocess_and_save(data_dir, stock_data):
    all_data = pd.concat(stock_data, axis=0)
    all_data['date'] = pd.to_datetime(all_data['date'])
    all_data.sort_values('date', ascending=True, inplace=True)
    
    # 按股票代码分组，确保每只股票的时间序列不被打乱
    all_data = all_data.groupby('stock_code').apply(lambda x: x.drop('stock_code', axis=1).sort_values('date')).reset_index(drop=True)
    
    # 划分数据集：50% 训练集，25% 验证集，25% 测试集
    train_data, test_data = train_test_split(all_data, test_size=0.5, shuffle=False)
    val_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=False)
    
    # 保存到CSV文件
    train_data.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    val_data.to_csv(os.path.join(data_dir, 'val_data.csv'), index=False)
    test_data.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
    
    print("✅ 数据已成功划分并保存：train_data.csv, val_data.csv, test_data.csv")

def main():
    data_dir = './training_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 登录
    login_baostock()
    
    # 下载HS300成分股列表
    hs300_stocks = fetch_hs300_stocks()
    stock_codes = hs300_stocks['code'].tolist()
    
    # 下载每支股票的数据
    stock_data = download_stock_data(stock_codes, data_dir)
    
    # 数据预处理与保存
    preprocess_and_save(data_dir, stock_data)

if __name__ == "__main__":
    main()
