import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from datetime import datetime

class QuantLabelerDataset(Dataset):
    def __init__(self, data_dir, label_dir, tokenizer, seq_len=60):
        """
        参数:
        - data_dir: 原始行情文件夹路径 (你的 'data/')
        - label_dir: 标注文件文件夹路径 (你的 'labels/')
        - tokenizer: Kronos 的分词器
        - seq_len: 序列长度 (默认 60)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = [] # 存储所有准备好的样本索引

        # 1. 扫描所有标注文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith("_labels.csv")]

        print(f"🔄 正在加载数据... 发现 {len(label_files)} 个标注文件")

        for l_file in label_files:
            # 解析品种名 (例如 "rb_labels.csv" -> "rb.csv")
            symbol_key = l_file.replace("_labels.csv", "")
            raw_file = f"{symbol_key}.csv"
            raw_path = os.path.join(data_dir, raw_file)
            label_path = os.path.join(label_dir, l_file)

            # 必须同时存在原始数据和标注数据
            if not os.path.exists(raw_path):
                continue

            # 2. 加载数据
            df_raw = pd.read_csv(raw_path)
            df_label = pd.read_csv(label_path)

            # 统一时间格式 (非常关键，防止字符串和datetime不匹配)
            # 假设你的 csv 里是字符串格式，我们统一转为 datetime 对象以便比较
            df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
            df_label['datetime'] = pd.to_datetime(df_label['datetime'])

            # 3. 匹配逻辑 (把标注点映射回原始 K 线)
            # 我们遍历每一个标注点
            for _, row in df_label.iterrows():
                target_time = row['datetime']
                label = int(row['label'])

                # 在原始数据中找到这一行
                # 使用 searchsorted 或直接 mask 查找 (数据量不大直接 mask 即可)
                # 找到 目标时间 对应的索引
                match = df_raw.index[df_raw['datetime'] == target_time].tolist()
                
                if not match:
                    continue # 标注的时间点在原始数据里找不到（可能重新下载了数据导致不匹配）
                
                idx = match[0]

                # 4. 截取前 60 根 (Sequence)
                # 如果前面的数据不够 60 根，就跳过
                if idx < seq_len:
                    continue
                
                # 截取范围: [idx - seq_len : idx] 
                # 注意：是否包含当前这根 K 线？通常意图识别是包含当前 K 线的
                # 这里的切片是 df_raw.iloc[idx - 59 : idx + 1]
                kline_segment = df_raw.iloc[idx - seq_len + 1 : idx + 1].copy()

                # 存入内存列表
                self.samples.append({
                    'df': kline_segment, # 这里存 DataFrame，取的时候再 Tokenize，省内存
                    'label': label,
                    'info': f"{symbol_key} @ {target_time}" # 方便调试
                })

        print(f"✅ 数据加载完成！共构建 {len(self.samples)} 个有效样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        df = item['df']
        label = item['label']

        # --- 调用 Kronos Tokenizer ---
        # 假设 tokenizer 接受 pandas dataframe
        # 如果 tokenizer 需要特定列名 (open, high, low, close, volume)，请确保 df 里有
        # 注意：这里需要根据 shiyu-coder 的官方 tokenizer API 调整
        try:
            # 假设 API 是这样的
            input_ids = self.tokenizer.encode(df) 
        except:
            # 这是一个占位符，防止你还没下载 tokenizer 代码报错
            # 实际跑的时候，tokenizer.encode 会返回 list 或 tensor
            input_ids = [0] * 60 

        # 转 Tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # 确保维度匹配 (Squeeze/Unsqueeze 根据模型要求)
        # 通常 Dataset 返回 (Seq_Len,)，DataLoader 会自动变成 (Batch, Seq_Len)
        return input_ids.squeeze(), torch.tensor(label, dtype=torch.long)

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟测试
    print("测试 Adapter...")
    # 假设你已经有了 tokenizer
    class MockTokenizer:
        def encode(self, df): return [1] * 60
    
    dataset = QuantLabelerDataset(
        data_dir="data", 
        label_dir="labels", 
        tokenizer=MockTokenizer()
    )
    
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"样本 0 输入形状: {x.shape}, 标签: {y}")
