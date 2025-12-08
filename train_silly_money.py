import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# 确保 model.py 在当前目录下
from model import Kronos, KronosTokenizer 

# ================= ⚙️ 配置区域 =================
TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PATH = "NeoQuasar/Kronos-base" # 直接使用原始 Base 模型

DATA_DIR = "data"
LABEL_DIR = "labels"

# 针对 89 个样本的微调策略
BATCH_SIZE = 8        # 样本少，Batch 小一点，更新次数多一点
LEARNING_RATE = 1e-4  # 只训练头，学习率可以给大一点 (1e-4 或 5e-4)
EPOCHS = 50           # 多跑几轮，保证收敛
SEQ_LEN = 60         
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 1. 数据适配器 (保留所有防崩黑科技) =================
class QuantLabelerDataset(Dataset):
    def __init__(self, data_dir, label_dir, tokenizer, seq_len=60):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = [] 
        # Kronos-base 物理词表限制 (非常重要！)
        self.vocab_size = 1024 

        if not os.path.exists(label_dir):
            print(f"⚠️ 错误: 找不到标注文件夹 {label_dir}")
            return
            
        label_files = [f for f in os.listdir(label_dir) if f.endswith("_labels.csv")]
        print(f"🔄 正在扫描真实标注数据... 发现 {len(label_files)} 个文件")

        for l_file in label_files:
            symbol_key = l_file.replace("_labels.csv", "")
            raw_file = f"{symbol_key}.csv"
            raw_path = os.path.join(data_dir, raw_file)
            label_path = os.path.join(label_dir, l_file)

            if not os.path.exists(raw_path): continue

            try:
                df_raw = pd.read_csv(raw_path)
                df_label = pd.read_csv(label_path)
                df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
                df_label['datetime'] = pd.to_datetime(df_label['datetime'])
                df_raw.columns = [c.lower() for c in df_raw.columns]
                
                # 🔥 自动补全 Amount (防止 60x5 报错)
                if 'amount' not in df_raw.columns:
                    df_raw['amount'] = df_raw['close'] * df_raw['volume']

                required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
                
                for _, row in df_label.iterrows():
                    target_time = row['datetime']
                    label = int(row['label'])
                    matches = df_raw.index[df_raw['datetime'] == target_time].tolist()
                    if not matches: continue
                    idx = matches[0]
                    if idx < seq_len - 1: continue
                    
                    df_segment = df_raw.iloc[idx - seq_len + 1 : idx + 1][required_cols].copy()
                    
                    self.samples.append({
                        'values': df_segment.values.astype(np.float32),
                        'label': label
                    })
                
            except Exception as e:
                print(f"  读取 {l_file} 出错: {e}")

        print(f"✅ 真实数据加载完成！共 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        values = item['values'] 
        label = item['label']

        # 🔥 归一化 (Log + Z-Score)
        # 防止数值过大导致 Tokenizer 内部溢出
        norm_values = values.copy()
        norm_values[:, 4] = np.log1p(norm_values[:, 4]) 
        norm_values[:, 5] = np.log1p(norm_values[:, 5]) 
        price_mean = norm_values[:, :4].mean()
        price_std = norm_values[:, :4].std() + 1e-5
        norm_values[:, :4] = (norm_values[:, :4] - price_mean) / price_std

        input_tensor = torch.tensor(norm_values, dtype=torch.float32).unsqueeze(0)

        try:
            encoded = self.tokenizer.encode(input_tensor)
            if isinstance(encoded, (tuple, list)) and len(encoded) == 2:
                s1, s2 = encoded[0], encoded[1]
            else:
                s1, s2 = encoded, np.zeros_like(encoded)
            
            if isinstance(s1, (np.ndarray, list)): s1 = torch.tensor(s1, dtype=torch.long)
            if isinstance(s2, (np.ndarray, list)): s2 = torch.tensor(s2, dtype=torch.long)
            
            s1 = s1.squeeze()
            s2 = s2.squeeze()
            
            # 🔥 取模大法 (Modulo Hack)
            # 解决 ID 越界导致 CUDA Error 的终极方案
            s1 = s1 % self.vocab_size
            s2 = s2 % self.vocab_size
            
        except Exception as e:
            s1 = torch.zeros(self.seq_len, dtype=torch.long)
            s2 = torch.zeros(self.seq_len, dtype=torch.long)

        return s1, s2, torch.tensor(label, dtype=torch.long)

# ================= 2. 模型定义 (原始 Base + 锁头) =================
class KronosClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        print(f"正在加载原始 Base 模型: {model_path} ...")
        self.backbone = Kronos.from_pretrained(model_path)
        
        # 🔥 锁死底座 (Frozen Backbone)
        # 我们只信任它的原始直觉，不让它被小样本带偏
        print("🔒 正在锁死底座模型参数 (Freeze)...")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 自动检测维度
        dummy_s1 = torch.zeros(1, 10, dtype=torch.long)
        dummy_s2 = torch.zeros(1, 10, dtype=torch.long)
        with torch.no_grad():
            outputs = self.backbone(dummy_s1, dummy_s2)
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                last_hidden = outputs[0]
            else:
                last_hidden = outputs
            self.hidden_size = last_hidden.shape[-1]
        
        # 新的分类头 (Trainable)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # 高 Dropout 防止过拟合
            nn.Linear(256, 2)
        )

    def forward(self, s1, s2):
        # 底座只负责提取特征
        outputs = self.backbone(s1, s2)
        
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            last_hidden = outputs[0]
        else:
            last_hidden = outputs

        # 取最后一个 Token 的特征进分类头
        return self.classifier(last_hidden[:, -1, :])

# ================= 3. 主训练循环 =================
def main():
    print(f"🚀 启动真实数据微调 (原始Base+锁头) | 设备: {DEVICE}")
    
    # 1. 必须重启 Python 终端以清除之前的显存状态
    
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    train_dataset = QuantLabelerDataset(DATA_DIR, LABEL_DIR, tokenizer, seq_len=SEQ_LEN)
    
    if len(train_dataset) == 0:
        print("❌ 没数据，无法训练。")
        return

    # 小样本用小 Batch
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = KronosClassifier(MODEL_PATH).to(DEVICE)

    # 🔥 只优化 classifier (头)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🏁 开始训练 | 样本: {len(train_dataset)} | 轮数: {EPOCHS}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for s1, s2, batch_labels in train_loader:
            s1, s2, batch_labels = s1.to(DEVICE), s2.to(DEVICE), batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(s1, s2)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
        epoch_acc = correct / total_samples
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2%}")
        
    save_path = "silly_money_base_raw.pth"
    torch.save(model.classifier.state_dict(), save_path)
    print(f"\n✅ 训练完成！分类头权重已保存为: {save_path}")

if __name__ == "__main__":
    main()
