import pandas as pd
import torch
import numpy as np
from model import KronosTokenizer

TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
TRAIN_DATA = "synthetic_data/train.npy"

def main():
    print("🔍 正在检查 Tokenizer 对合成数据的处理 (修复版)...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # 加载一条数据
    data = np.load(TRAIN_DATA, allow_pickle=True)
    sample = data[0]
    # 获取 numpy array (60, 5)
    values = sample['df'] 
    
    print(f"\n--- 原始数据形状: {values.shape} ---")
    print(values[:5]) # 打印前5行看看数值
    
    # --- 关键修复：手动转 Tensor ---
    # 1. 转为 Tensor
    # 2. 转为 float32 (神经网络只吃 float32)
    # 3. 增加 batch 维度: [60, 5] -> [1, 60, 5]
    input_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    
    print(f"\n--- 输入 Tensor 形状: {input_tensor.shape} ---")
    
    # Tokenize
    try:
        encoded = tokenizer.encode(input_tensor)
        
        if isinstance(encoded, (tuple, list)) and len(encoded) == 2:
            s1, s2 = encoded[0], encoded[1]
        else:
            s1 = encoded
            
        # 移除 batch 维度方便查看
        s1 = torch.tensor(s1).squeeze()
        
        print("\n--- Tokenizer 输出 (S1 IDs) ---")
        print(s1)
        
        # 统计唯一值
        unique_tokens = torch.unique(s1)
        print(f"\n📊 唯一 Token 数量: {len(unique_tokens)}")
        
        if len(unique_tokens) < 5:
            print("❌ Tokenizer 输出太单一，合成数据可能还是有问题。")
        else:
            print("✅ Tokenizer 工作正常！确实输出了多样化的 ID。")
            print("🎉 结论：之前的训练失败是因为输入格式不对，模型一直在吃‘0’。")
            
    except Exception as e:
        print(f"❌ 依然报错: {e}")

if __name__ == "__main__":
    main()
