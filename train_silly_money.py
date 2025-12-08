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

# ================= 配置区域 =================
TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PATH = "NeoQuasar/Kronos-small" 

DATA_DIR = "data"
LABEL_DIR = "labels"

BATCH_SIZE = 8       
LEARNING_RATE = 1e-4 
EPOCHS = 10          
SEQ_LEN = 60         
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 1. 数据适配器 (修复双流输出) =================
class QuantLabelerDataset(Dataset):
    def __init__(self, data_dir, label_dir, tokenizer, seq_len=60):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = [] 

        if not os.path.exists(label_dir):
            print(f"⚠️ 错误: 找不到标注文件夹 {label_dir}")
            return
            
        label_files = [f for f in os.listdir(label_dir) if f.endswith("_labels.csv")]
        print(f"🔄 正在扫描数据... 发现 {len(label_files)} 个标注文件")

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
                
                # 预先提取需要的列，确保列名小写
                df_raw.columns = [c.lower() for c in df_raw.columns]
                
                for _, row in df_label.iterrows():
                    target_time = row['datetime']
                    label = int(row['label'])
                    matches = df_raw.index[df_raw['datetime'] == target_time].tolist()
                    if not matches: continue
                    idx = matches[0]
                    if idx < seq_len - 1: continue
                    
                    # 截取
                    df_segment = df_raw.iloc[idx - seq_len + 1 : idx + 1].copy()
                    
                    self.samples.append({
                        'df': df_segment,
                        'label': label
                    })
                
            except Exception as e:
                print(f"  读取 {l_file} 出错: {e}")

        print(f"✅ 数据加载完成！共构建 {len(self.samples)} 个有效样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        df = item['df']
        label = item['label']

        try:
            # --- 修复: 正确处理 Tokenizer 的双重输出 (s1, s2) ---
            encoded = self.tokenizer.encode(df)
            
            # 强制解包 tuple/list
            if isinstance(encoded, (tuple, list)) and len(encoded) == 2:
                s1_ids = encoded[0]
                s2_ids = encoded[1]
            else:
                # 容错处理
                s1_ids = encoded
                s2_ids = np.zeros_like(encoded)

            # 转 Tensor
            if isinstance(s1_ids, (np.ndarray, list)):
                s1_ids = torch.tensor(s1_ids, dtype=torch.long)
            if isinstance(s2_ids, (np.ndarray, list)):
                s2_ids = torch.tensor(s2_ids, dtype=torch.long)
            
            s1_ids = s1_ids.squeeze()
            s2_ids = s2_ids.squeeze()
            
        except Exception as e:
            # print(f"Tokenizer error: {e}")
            s1_ids = torch.zeros(self.seq_len, dtype=torch.long)
            s2_ids = torch.zeros(self.seq_len, dtype=torch.long)

        return s1_ids, s2_ids, torch.tensor(label, dtype=torch.long)

# ================= 2. 模型定义 (修复维度与输入) =================
class KronosClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        print(f"正在加载预训练模型: {model_path} ...")
        self.backbone = Kronos.from_pretrained(model_path)
        
        # 冻结参数
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # --- 修复: 自动检测隐藏层维度 ---
        print("🔍 正在自动检测模型输出维度...")
        dummy_s1 = torch.zeros(1, 10, dtype=torch.long) # 构造假数据
        dummy_s2 = torch.zeros(1, 10, dtype=torch.long)
        
        with torch.no_grad():
            try:
                outputs = self.backbone(dummy_s1, dummy_s2)
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden = outputs.last_hidden_state
                elif isinstance(outputs, tuple):
                    last_hidden = outputs[0]
                else:
                    last_hidden = outputs
                
                self.hidden_size = last_hidden.shape[-1]
                print(f"✅ 检测成功! Hidden Size = {self.hidden_size}")
            except Exception as e:
                print(f"⚠️ 检测失败 ({e}), 回退到默认 768")
                self.hidden_size = 768
        
        # 定义分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, s1_ids, s2_ids):
        # --- 修复: 传入双流参数 ---
        outputs = self.backbone(s1_ids, s2_ids)
        
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs

        # 取最后一个 Token
        last_token_feature = last_hidden_state[:, -1, :]
        logits = self.classifier(last_token_feature)
        return logits

# ================= 3. 主训练循环 =================
def main():
    print(f"🚀 启动训练任务 | 设备: {DEVICE}")
    
    print("正在加载 Tokenizer...")
    try:
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
        print("✅ Tokenizer 加载成功")
    except Exception as e:
        print(f"❌ Tokenizer 加载失败: {e}")
        return

    train_dataset = QuantLabelerDataset(DATA_DIR, LABEL_DIR, tokenizer, seq_len=SEQ_LEN)
    
    if len(train_dataset) == 0:
        print("❌ 没有找到有效样本")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = KronosClassifier(MODEL_PATH).to(DEVICE)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🏁 开始训练 | 样本数: {len(train_dataset)} | 轮数: {EPOCHS}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # --- 修复: 解包 s1, s2, label ---
        for s1_ids, s2_ids, batch_labels in progress_bar:
            s1_ids = s1_ids.to(DEVICE)
            s2_ids = s2_ids.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 传入双流
            logits = model(s1_ids, s2_ids)
            
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
            acc_str = f"{correct/total_samples:.2%}" if total_samples > 0 else "0.00%"
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': acc_str})
        
    torch.save(model.classifier.state_dict(), "silly_money_head.pth")
    print(f"\n✅ 训练完成！分类头已保存为 silly_money_head.pth")

if __name__ == "__main__":
    main()
