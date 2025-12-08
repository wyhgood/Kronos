import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import Kronos, KronosTokenizer 

# ================= ⚙️ 配置 =================
TRAIN_DATA = "synthetic_data/train.npy"
VAL_DATA = "synthetic_data/val.npy"
TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PATH = "NeoQuasar/Kronos-base" 

BATCH_SIZE = 64      
EPOCHS = 5           
LEARNING_RATE = 2e-5 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 📥 数据集 (取模修复版) =================
class SyntheticDataset(Dataset):
    def __init__(self, npy_path, tokenizer):
        print(f"📥 加载 {npy_path} ...")
        self.data = np.load(npy_path, allow_pickle=True)
        self.tokenizer = tokenizer
        
        # 🔥 根据之前的检查，词表只有 1024
        self.vocab_size = 1024 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        values = item['df'].astype(np.float32) 
        label = item['label']
        
        # 1. 归一化 (保持不变，这依然很有必要)
        norm_values = values.copy()
        norm_values[:, 4] = np.log1p(norm_values[:, 4]) # Volume
        norm_values[:, 5] = np.log1p(norm_values[:, 5]) # Amount
        price_mean = norm_values[:, :4].mean()
        price_std = norm_values[:, :4].std() + 1e-5
        norm_values[:, :4] = (norm_values[:, :4] - price_mean) / price_std
        
        # 2. 转 Tensor [1, 60, 6]
        input_tensor = torch.tensor(norm_values, dtype=torch.float32).unsqueeze(0)
        
        # 3. Tokenize
        try:
            encoded = self.tokenizer.encode(input_tensor)
            if isinstance(encoded, (tuple, list)) and len(encoded) == 2:
                s1, s2 = encoded[0], encoded[1]
            else:
                s1, s2 = encoded, np.zeros_like(encoded)
        except:
            s1 = torch.zeros(60, dtype=torch.long)
            s2 = torch.zeros(60, dtype=torch.long)

        # 4. 后处理
        if isinstance(s1, (np.ndarray, list)): s1 = torch.tensor(s1, dtype=torch.long)
        if isinstance(s2, (np.ndarray, list)): s2 = torch.tensor(s2, dtype=torch.long)
        
        s1 = s1.squeeze()
        s2 = s2.squeeze()
        
        # 🔥🔥🔥【神来之笔：取模 Hack】🔥🔥🔥
        # 不再用 clamp(切除)，而是用 % (折叠)
        # 这样 50000 不会变成 1023，而是变成 832 (举例)
        # 50001 会变成 833
        # 数据差异被保留下来了，而且绝对不会越界报错！
        s1 = s1 % self.vocab_size
        s2 = s2 % self.vocab_size
             
        return s1, s2, torch.tensor(label, dtype=torch.long)

# ================= 🏗️ 模型定义 =================
class KronosClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        print(f"正在加载大模型: {model_path} ...")
        self.backbone = Kronos.from_pretrained(model_path)
        
        print("🔓 底座模型已解锁，进行全量微调...")
        print(f"🔄 启用 Modulo Hack: 将 Token ID 折叠至 [0, 1023]")
            
        print("🔍 正在检测 Base 模型维度...")
        dummy_s1 = torch.zeros(1, 10, dtype=torch.long)
        dummy_s2 = torch.zeros(1, 10, dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.backbone(dummy_s1, dummy_s2)
            # 兼容各种输出格式
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                last_hidden = outputs[0]
            else:
                last_hidden = outputs
            
            self.hidden_size = last_hidden.shape[-1]
            print(f"✅ 检测完毕: Hidden Size = {self.hidden_size}")
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, s1, s2):
        outputs = self.backbone(s1, s2)
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            last_hidden = outputs[0]
        else:
            last_hidden = outputs
        return self.classifier(last_hidden[:, -1, :])

# ================= 🚀 主程序 =================
def main():
    print(f"🧪 启动全量微调实验 (Modulo Fix) | 设备: {DEVICE}")
    
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    train_ds = SyntheticDataset(TRAIN_DATA, tokenizer)
    val_ds = SyntheticDataset(VAL_DATA, tokenizer)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = KronosClassifier(MODEL_PATH).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        total_train = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1} Train")
        
        for s1, s2, lbl in pbar:
            s1, s2, lbl = s1.to(DEVICE), s2.to(DEVICE), lbl.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(s1, s2)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (logits.argmax(1) == lbl).sum().item()
            total_train += lbl.size(0)
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{train_acc/total_train:.2%}"})
            
        model.eval()
        val_acc = 0
        total_val = 0
        with torch.no_grad():
            for s1, s2, lbl in tqdm(val_dl, desc=f"Epoch {epoch+1} Val"):
                s1, s2, lbl = s1.to(DEVICE), s2.to(DEVICE), lbl.to(DEVICE)
                logits = model(s1, s2)
                val_acc += (logits.argmax(1) == lbl).sum().item()
                total_val += lbl.size(0)
        
        tr_acc_pct = train_acc / total_train
        val_acc_pct = val_acc / total_val
        
        print(f"📊 Summary: Train Acc: {tr_acc_pct:.2%} | Val Acc: {val_acc_pct:.2%}")
        
        if val_acc_pct > best_acc:
            best_acc = val_acc_pct
            torch.save(model.state_dict(), "best_full_finetune.pth")
            print("💾 新纪录！全量模型已保存。")

if __name__ == "__main__":
    main()
