import torch
import torch.nn as nn
import numpy as np
import os
import sys

# ================= ⚙️ 配置 =================
DATA_DIR = "data_double_top_v1"
MODEL_PATH = "double_top_expert.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 60


# verify_logic.py 配置修改
# -------------------------------------------------
DATA_DIR = "data_complex_ohlc_ema"
MODEL_PATH = "kronos_ema_expert.pth"

# 确保模型实例化也是 6 维
# model = Kronos(input_dim=6).to(DEVICE) 
# -------------------------------------------------
# ================= 🧠 模型定义 (必须与训练时完全一致) =================
# 为了防止 Import 错误，我们直接把训练用的类定义贴在这里
class Kronos(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # Input embedding: maps 6 features to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output 2 classes
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        
        # Take the last time step
        last_step_feature = x[:, -1, :] 
        return self.fc(last_step_feature)

# ================= 🔍 验证逻辑 =================
def verify():
    print("🧐 正在进行逻辑验收 (独立定义版)...")

    # 1. 加载数据
    x_path = os.path.join(DATA_DIR, "X_test.npy")
    t_path = os.path.join(DATA_DIR, "test_types.npy")

    if not os.path.exists(x_path):
        print(f"❌ 找不到测试数据: {x_path}")
        return

    X_test = np.load(x_path).astype(np.float32)
    types = np.load(t_path)    # 1=Pos, 2=HardNeg, 3=EasyNeg

    print(f"📊 加载测试集: {X_test.shape}")

    # 2. 归一化 (Z-Score, 与训练一致)
    print("🔄 正在归一化...")
    for i in range(len(X_test)):
        mean = np.mean(X_test[i], axis=0)
        std = np.std(X_test[i], axis=0) + 1e-6
        X_test[i] = (X_test[i] - mean) / std

    # 3. 加载模型
    print("🧠 加载模型权重...")
    model = Kronos(input_dim=6).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 提示: 可能是权重文件损坏，或者是旧的模型结构遗留。")
        return
        
    model.eval()

    # 4. 预测
    print("⚡ 正在推理...")
    preds = []
    batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_input = torch.tensor(X_test[i : i+batch_size]).to(DEVICE)
            logits = model(batch_input)
            # 获取类别 (0 或 1)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(batch_preds)
            
    preds = np.concatenate(preds)

    # --- 细分分析 ---
    print("\n📊 详细测试报告:")
    print("-" * 30)

    # 1. 正样本 (Positive)
    idx_pos = (types == 1)
    if np.sum(idx_pos) > 0:
        acc_pos = np.mean(preds[idx_pos] == 1)
        print(f"✅ 正样本 (进场信号) 捕捉率: {acc_pos:.2%} (目标 > 95%)")
    else:
        print("⚠️ 无正样本")

    # 2. 困难负样本 (Hard Negative)
    idx_hard = (types == 2)
    if np.sum(idx_hard) > 0:
        err_hard = np.mean(preds[idx_hard] == 1) 
        print(f"⚠️ 困难负样本 (假动作) 误判率: {err_hard:.2%} (越低越好，目标 < 5%)")
    else:
        print("⚠️ 无困难负样本")

    # 3. 简单负样本 (Easy Negative)
    idx_easy = (types == 3)
    if np.sum(idx_easy) > 0:
        err_easy = np.mean(preds[idx_easy] == 1)
        print(f"⛔ 简单负样本 (真突破) 误判率: {err_easy:.2%}")
    else:
        print("⚠️ 无简单负样本")

    print("-" * 30)
    
    # 最终结论
    success = True
    if np.sum(idx_pos) > 0 and acc_pos < 0.95: success = False
    if np.sum(idx_hard) > 0 and err_hard > 0.05: success = False
    
    if success:
        print("🎉 结论: 完美！模型成功学会了区分‘入场K线’和‘假动作’。")
    else:
        print("🤔 结论: 还需要微调。")

if __name__ == "__main__":
    verify()
