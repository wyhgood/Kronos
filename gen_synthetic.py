import matplotlib
matplotlib.use('Agg') 
import numpy as np
import pandas as pd
import os
import mplfinance as mpf
from tqdm import tqdm

# ================= ⚙️ 配置 =================
NUM_POS = 30000   # 1:1 平衡
NUM_NEG = 30000   
SEQ_LEN = 60      
SAVE_DIR = "synthetic_data"
IMG_DIR = "synthetic_verify" 

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)

# ================= 🎨 造假工坊 (升级版: 含 Amount) =================

def create_candle_from_price(price_seq):
    noise_high = np.abs(np.random.normal(0, 0.2, size=len(price_seq)))
    noise_low = np.abs(np.random.normal(0, 0.2, size=len(price_seq)))
    
    open_p = np.roll(price_seq, 1)
    open_p[0] = price_seq[0] * (1 + np.random.normal(0, 0.005))
    open_p += np.random.normal(0, 0.1, size=len(price_seq))
    
    close_p = price_seq
    high_p = np.maximum(open_p, close_p) + noise_high
    low_p = np.minimum(open_p, close_p) - noise_low
    volume = np.random.randint(100, 1000, size=len(price_seq)).astype(float)
    
    # 🔥 【关键修复】生成 Amount (成交额 ≈ 收盘价 * 成交量)
    amount = close_p * volume 
    
    df = pd.DataFrame({
        'open': open_p, 
        'high': high_p, 
        'low': low_p, 
        'close': close_p, 
        'volume': volume,
        'amount': amount  # <--- 新增第 6 列
    })
    
    df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='min')
    return df

def generate_double_top():
    """生成双顶"""
    noise_level = np.random.uniform(0.1, 0.3)
    p1 = np.random.randint(12, 18)  
    p2 = np.random.randint(28, 32)  
    p3 = np.random.randint(42, 48)  
    
    skeleton = np.zeros(SEQ_LEN)
    skeleton[:p1] = np.linspace(100, 110, p1)
    skeleton[p1:p2] = np.linspace(110, 103, p2-p1)
    skeleton[p2:p3] = np.linspace(103, 110.5, p3-p2) 
    skeleton[p3:] = np.linspace(110.5, 95, SEQ_LEN-p3)
    
    noise = np.cumsum(np.random.normal(0, noise_level, SEQ_LEN))
    price = skeleton + noise
    return create_candle_from_price(price)

def generate_negative_sample():
    """生成非双顶"""
    mode = np.random.randint(0, 3)
    base = 100
    noise = np.cumsum(np.random.normal(0, 0.5, SEQ_LEN))
    
    if mode == 0: price = base + noise
    elif mode == 1: price = base + np.linspace(0, 20, SEQ_LEN) + noise
    else: price = base + np.linspace(0, -20, SEQ_LEN) + noise
        
    return create_candle_from_price(price)

# ================= 🚀 执行 =================
def main():
    print(f"🏭 开始制造数据 (6列完整版)... 正: {NUM_POS} | 负: {NUM_NEG}")
    
    all_data = []
    
    # 1. 生成正样本
    print("正在生成双顶 (Label 1)...")
    for i in tqdm(range(NUM_POS)):
        df = generate_double_top()
        all_data.append({'df': df.values, 'label': 1}) 
        if i < 5: 
            mc = mpf.make_marketcolors(up='r', down='g', volume='in')
            s = mpf.make_mpf_style(marketcolors=mc)
            mpf.plot(df, type='candle', style=s, volume=True, savefig=f"{IMG_DIR}/pos_{i}.png")

    # 2. 生成负样本
    print("正在生成负样本 (Label 0)...")
    for i in tqdm(range(NUM_NEG)):
        df = generate_negative_sample()
        all_data.append({'df': df.values, 'label': 0})

    # 3. 打乱与保存
    np.random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    np.save(f"{SAVE_DIR}/train.npy", train_data)
    np.save(f"{SAVE_DIR}/val.npy", val_data)
    
    print(f"✅ 数据重造完毕! 每个样本包含 6 列特征。")

if __name__ == "__main__":
    main()
