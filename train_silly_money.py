import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm # 进度条库，如果没有请 pip install tqdm

# ================= 配置区域 =================
# 模型路径 (请确保这是你 download_weights.py 下载的路径)
MODEL_PATH = "./checkpoints/Kronos-small"

# 数据路径 (对应你的 Streamlit 目录结构)
DATA_DIR = "data"
LABEL_DIR = "labels"

# 训练参数
BATCH_SIZE = 8       # 如果显存不够，改小一点，比如 4 或 2
LEARNING_RATE = 1e-4 # 微调通常用较小的学习率
EPOCHS = 10          # 训练轮数
SEQ_LEN = 60         # 回看窗口长度 (和你的逻辑一致)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 1. 数据适配器 (直接对接你的标注工具) =================
class QuantLabelerDataset(Dataset):
    def __init__(self, data_dir, label_dir, tokenizer, seq_len=60):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = [] 

        # 扫描所有标注文件
        if not os.path.exists(label_dir):
            print(f"⚠️ 错误: 找不到标注文件夹 {label_dir}")
            return
            
        label_files = [f for f in os.listdir(label_dir) if f.endswith("_labels.csv")]
        print(f"🔄 正在扫描数据... 发现 {len(label_files)} 个标注文件")

        for l_file in label_files:
            # 解析品种名 (例如 "rb_labels.csv" -> "rb.csv")
            symbol_key = l_file.replace("_labels.csv", "")
            raw_file = f"{symbol_key}.csv"
            raw_path = os.path.join(data_dir, raw_file)
            label_path = os.path.join(label_dir, l_file)

            if not os.path.exists(raw_path):
                print(f"  跳过 {l_file}: 找不到对应的原始行情文件 {raw_file}")
                continue

            # 加载数据
            try:
                df_raw = pd.read_csv(raw_path)
                df_label = pd.read_csv(label_path)
                
                # 统一时间格式
                df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
                df_label['datetime'] = pd.to_datetime(df_label['datetime'])
                
                # 遍历标注点
                count = 0
                for _, row in df_label.iterrows():
                    target_time = row['datetime']
                    label = int(row['label'])

                    # 查找对应的时间索引
                    matches = df_raw.index[df_raw['datetime'] == target_time].tolist()
                    if not matches: continue
                    
                    idx = matches[0]

                    # 确保前面有足够的数据 (60根)
                    if idx < seq_len - 1: continue
                    
                    # 截取 DataFrame片段 (包含当前根)
                    # 范围: [idx - 59, idx] 共 60 根
                    df_segment = df_raw.iloc[idx - seq_len + 1 : idx + 1].copy()
                    
                    self.samples.append({
                        'df': df_segment,
                        'label': label
                    })
                    count += 1
                # print(f"  {symbol_key}: 加载了 {count} 个样本")
                
            except Exception as e:
                print(f"  读取 {l_file} 出错: {e}")

        print(f"✅ 数据加载完成！共构建 {len(self.samples)} 个有效样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        df = item['df']
        label = item['label']

        # --- 调用 Kronos Tokenizer ---
        # 注意: Kronos Tokenizer 通常需要 pandas DataFrame 作为输入
        # 它会自动识别 open, high, low, close, volume 列
        try:
            # 官方 API 调用方式
            input_ids = self.tokenizer.encode(df)
            
            # 如果 tokenizer 返回的是 list，转 tensor
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            elif isinstance(input_ids, np.ndarray):
                input_ids = torch.from_numpy(input_ids).long()
                
            # 确保去掉多余的 batch 维度 (如果有)
            input_ids = input_ids.squeeze()
            
        except Exception as e:
            print(f"Tokenizer 编码错误: {e}")
            # 出错时返回一个全0的tensor防止程序崩溃
            input_ids = torch.zeros(self.seq_len, dtype=torch.long)

        return input_ids, torch.tensor(label, dtype=torch.long)

# ================= 2. 模型定义 (冻结骨架 + 分类头) =================
class KronosClassifier(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        print(f"正在加载预训练模型: {pretrained_path} ...")
        
        # 加载底座 (Transformer)
        self.backbone = AutoModel.from_pretrained(
            pretrained_path, 
            trust_remote_code=True
        )
        
        # --- 冻结底座参数 (Linear Probing) ---
        # 这样我们只训练分类头，保护底座的“通识”
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 获取隐藏层维度 (Kronos-small 通常是 768，Base 是 1024)
        # 尝试从 config 读取，读不到就默认 768
        try:
            self.hidden_size = self.backbone.config.hidden_size
        except:
            self.hidden_size = 768 
            
        print(f"模型加载成功，隐藏层维度: {self.hidden_size}，底座已冻结。")
        
        # --- 定义分类头 ---
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),     # 防止过拟合
            nn.Linear(256, 2)    # 输出: [No_Intent, Yes_Intent]
        )

    def forward(self, input_ids):
        # Kronos 是 decoder-only，输入 input_ids 即可
        # output.last_hidden_state 形状: [batch, seq_len, hidden]
        outputs = self.backbone(input_ids=input_ids)
        
        # 我们只取序列的最后一个 Token 的特征
        # 因为在自回归模型中，最后一个 Token 包含了整个序列的信息
        last_token_feature = outputs.last_hidden_state[:, -1, :]
        
        # 过分类头
        logits = self.classifier(last_token_feature)
        return logits

# ================= 3. 主训练循环 =================
def main():
    print(f"🚀 启动训练任务 | 设备: {DEVICE}")
    
    # A. 加载 Tokenizer
    print("正在加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Tokenizer 加载失败: {e}")
        print("请检查 MODEL_PATH 是否正确，或者网络是否通畅。")
        return

    # B. 准备数据集
    train_dataset = QuantLabelerDataset(DATA_DIR, LABEL_DIR, tokenizer, seq_len=SEQ_LEN)
    
    if len(train_dataset) == 0:
        print("❌ 没有找到有效样本，请先去 Streamlit 标注一些数据！")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # C. 初始化模型
    model = KronosClassifier(MODEL_PATH).to(DEVICE)

    # D. 定义优化器和损失函数
    # 只优化 classifier 的参数！
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # 自动处理 Softmax

    # E. 开始训练
    print(f"\n🏁 开始训练 | 样本数: {len(train_dataset)} | 批次大小: {BATCH_SIZE} | 轮数: {EPOCHS}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_ids, batch_labels in progress_bar:
            batch_ids = batch_ids.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            # 前向传播
            optimizer.zero_grad()
            logits = model(batch_ids)
            
            # 计算损失
            loss = criterion(logits, batch_labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
            # 更新进度条
            current_acc = correct / total_samples
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{current_acc:.2%}"})
        
        avg_loss = total_loss / len(train_loader)
        epoch_acc = correct / total_samples
        print(f"📊 Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.4f} | 准确率: {epoch_acc:.2%}")

    # F. 保存模型 (只保存分类头，因为底座没变，这样文件很小)
    save_path = "silly_money_head.pth"
    torch.save(model.classifier.state_dict(), save_path)
    print(f"\n✅ 训练完成！分类头已保存为: {save_path}")
    print("💡 提示: 推理时，加载底座后，再用 load_state_dict 加载这个文件即可。")

if __name__ == "__main__":
    main()
