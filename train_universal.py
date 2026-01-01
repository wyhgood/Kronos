import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys

# ================= ⚙️ 核心配置区域 (只改这里) =================

# 1. 任务模式
# "SCRATCH": 从零开始训练 (用于跑大量合成数据，如双顶实验)
# "FINETUNE": 加载已有模型微调 (用于跑真实数据，或者进阶合成数据)
TRAIN_MODE = "SCRATCH" 

# 2. 路径配置
DATA_DIR = "data_double_top_v1"   # 数据集文件夹
MODEL_SAVE_NAME = "kronos_model.pth" # 保存的模型名字
PRETRAINED_PATH = "double_top_expert.pth" # 如果是 FINETUNE 模式，读取哪个模型？

# 3. 训练超参数
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
SEQ_LEN = 60
INPUT_DIM = 6   # OHLC + Vol + Amt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# train_universal.py 头部修改
# train_universal.py 头部修改

# 指向新数据
DATA_DIR = "data_complex_ohlc_ema"
MODEL_SAVE_NAME = "kronos_ema_expert.pth"

# 🔥 核心修改：输入维度 = 6 (OHLC + 2个EMA)
INPUT_DIM = 6
# ================= 🧠 模型定义 (标准版) =================
# 这是一个标准的 Transformer 架构，兼容两种模式
class Kronos(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # 数值嵌入层 (Float -> Vector)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码 (学习型)
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        
        # Transformer 主干
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头 (2分类: 不进/进)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        # 取最后一个时间步特征
        return self.fc(x[:, -1, :])

# ================= 💾 数据加载器 (通用版) =================
# 统一使用 .npy 格式，无论是真实数据还是合成数据，先转成 npy 再喂进来
class UniversalDataset(Dataset):
    def __init__(self, x_path, y_path):
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"❌ 找不到数据文件: {x_path}")
            
        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.longlong) # Label必须是long
        
        # 🔥 自动 Z-Score 归一化
        print(f"🔄 正在归一化 {len(self.X)} 条数据...")
        for i in range(len(self.X)):
            mean = np.mean(self.X[i], axis=0)
            std = np.std(self.X[i], axis=0) + 1e-6
            self.X[i] = (self.X[i] - mean) / std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ================= 🚀 主程序 =================
def main():
    print(f"🚀 启动通用训练引擎 | 模式: {TRAIN_MODE} | 设备: {DEVICE}")
    
    # 1. 准备数据
    print(f"📂 读取数据: {DATA_DIR}")
    try:
        train_ds = UniversalDataset(
            os.path.join(DATA_DIR, "X_train.npy"), 
            os.path.join(DATA_DIR, "y_train.npy")
        )
        # 如果有测试集就加载，没有就跳过
        test_path = os.path.join(DATA_DIR, "X_test.npy")
        if os.path.exists(test_path):
            test_ds = UniversalDataset(
                os.path.join(DATA_DIR, "X_test.npy"), 
                os.path.join(DATA_DIR, "y_test.npy")
            )
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        else:
            test_loader = None
            
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 初始化模型
    model = Kronos(input_dim=INPUT_DIM).to(DEVICE)
    
    # 3. 权重加载逻辑 (核心差异处理)
    if TRAIN_MODE == "FINETUNE":
        if os.path.exists(PRETRAINED_PATH):
            print(f"🧠 加载预训练权重: {PRETRAINED_PATH}")
            model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
            
            # 可选：如果是微调极少量数据，可以冻结主干
            # print("🔒 冻结 Transformer 主干...")
            # for param in model.transformer.parameters():
            #     param.requires_grad = False
        else:
            print(f"⚠️ 警告: 找不到预训练模型 {PRETRAINED_PATH}，将从零开始！")
    else:
        print("✨ 初始化全新模型 (From Scratch)...")

    # 4. 优化器与损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环
    print(f"\n🏁 开始训练 | 轮数: {EPOCHS}")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # 测试集验证 (如果有)
        test_log = ""
        if test_loader:
            model.eval()
            t_correct = 0
            t_total = 0
            with torch.no_grad():
                for X_t, y_t in test_loader:
                    X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
                    out = model(X_t)
                    t_correct += (torch.argmax(out, dim=1) == y_t).sum().item()
                    t_total += y_t.size(0)
            test_acc = t_correct / t_total
            test_log = f"| Val Acc: {test_acc:.2%}"
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), MODEL_SAVE_NAME)
                test_log += " ⭐"
        else:
            # 如果没有测试集，每轮都保存
            torch.save(model.state_dict(), MODEL_SAVE_NAME)

        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2%} {test_log}")

    print(f"\n✅ 训练结束。模型已保存为: {MODEL_SAVE_NAME}")

if __name__ == "__main__":
    main()
