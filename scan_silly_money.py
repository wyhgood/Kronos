import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import Kronos, KronosTokenizer

# ================= ⚙️ 配置区域 (已修改) =================
TARGET_FILE = "data/doupo.csv"  # 你的目标文件
TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PATH = "NeoQuasar/Kronos-small"
WEIGHTS_PATH = "silly_money_head.pth"

SEQ_LEN = 60
STRIDE = 1                 # <--- 修改点：每根K线都算，不跳过
CONFIDENCE_THRESHOLD = 0.55 # <--- 修改点：阈值大幅降低，捕捉更多信号
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 🏗️ 模型定义 =================
class KronosClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        print(f"正在加载预训练模型: {model_path} ...")
        self.backbone = Kronos.from_pretrained(model_path)
        
        # 冻结参数
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 自动检测隐藏层维度
        print("🔍 正在自动检测模型输出维度...")
        dummy_s1 = torch.zeros(1, 10, dtype=torch.long)
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
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, s1_ids, s2_ids):
        outputs = self.backbone(s1_ids, s2_ids)
        
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs
            
        last_token_feature = last_hidden_state[:, -1, :]
        logits = self.classifier(last_token_feature)
        return logits

# ================= 📥 数据集定义 =================
class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, seq_len=60, stride=1):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []
        
        # 预处理
        self.df_values = df[['open', 'high', 'low', 'close', 'volume']].copy()
        self.meta_data = df[['datetime', 'close']].copy()
        
        # 生成索引 (stride=1 代表全覆盖)
        indices = range(seq_len, len(df), stride)
        print(f"🔪 正在切片... (步长: {stride}, 预计生成 {len(indices)} 个样本)")
        
        for i in indices:
            self.samples.append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        end_idx = self.samples[idx]
        start_idx = end_idx - self.seq_len
        
        df_segment = self.df_values.iloc[start_idx : end_idx]
        target_time = self.meta_data.iloc[end_idx-1]['datetime']
        target_price = self.meta_data.iloc[end_idx-1]['close']
        
        try:
            encoded = self.tokenizer.encode(df_segment)
            if isinstance(encoded, (tuple, list)) and len(encoded) == 2:
                s1_ids = encoded[0]
                s2_ids = encoded[1]
            else:
                s1_ids = encoded
                s2_ids = np.zeros_like(encoded)

            if isinstance(s1_ids, (np.ndarray, list)):
                s1_ids = torch.tensor(s1_ids, dtype=torch.long)
            if isinstance(s2_ids, (np.ndarray, list)):
                s2_ids = torch.tensor(s2_ids, dtype=torch.long)
                
            s1_ids = s1_ids.squeeze()
            s2_ids = s2_ids.squeeze()
        except Exception as e:
            s1_ids = torch.zeros(self.seq_len, dtype=torch.long)
            s2_ids = torch.zeros(self.seq_len, dtype=torch.long)
            
        return s1_ids, s2_ids, str(target_time), float(target_price)

# ================= 🚀 主程序 =================
def main():
    print(f"🕵️‍♀️ 启动 AI 全局扫描 (高灵敏度模式) | 目标: {TARGET_FILE}")
    
    if not os.path.exists(TARGET_FILE):
        print("❌ 文件不存在")
        return

    base_name = os.path.basename(TARGET_FILE).replace(".csv", "")
    ai_data_name = f"{base_name}_AI"
    ai_data_file = os.path.join("data", f"{ai_data_name}.csv")
    ai_label_file = os.path.join("labels", f"{ai_data_name}_labels.csv")

    print("⚙️ 加载模型中...")
    try:
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
        model = KronosClassifier(MODEL_PATH).to(DEVICE)
        
        if os.path.exists(WEIGHTS_PATH):
            state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            model.classifier.load_state_dict(state_dict)
            print("✅ 权重加载成功")
        else:
            print("❌ 找不到权重文件，请先训练！")
            return
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    model.eval()

    df = pd.read_csv(TARGET_FILE)
    df.columns = [c.lower() for c in df.columns]
    
    # 构造数据集 (stride=1)
    dataset = InferenceDataset(df, tokenizer, seq_len=SEQ_LEN, stride=STRIDE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    results = []
    print(f"🚀 开始扫描 {len(dataset)} 个窗口... (阈值: {CONFIDENCE_THRESHOLD})")
    
    with torch.no_grad():
        for s1_ids, s2_ids, batch_times, batch_prices in tqdm(dataloader):
            s1_ids = s1_ids.to(DEVICE)
            s2_ids = s2_ids.to(DEVICE)
            
            logits = model(s1_ids, s2_ids)
            probs = torch.softmax(logits, dim=1)
            pos_probs = probs[:, 1].cpu().numpy()
            
            for i, prob in enumerate(pos_probs):
                if prob > CONFIDENCE_THRESHOLD:
                    results.append({
                        'datetime': batch_times[i],
                        'label': 1,
                        'price': batch_prices[i].item() if isinstance(batch_prices[i], torch.Tensor) else batch_prices[i],
                        'confidence': prob
                    })
    
    print(f"✅ 扫描完成！发现 {len(results)} 个潜在机会。")
    
    if len(results) > 0:
        shutil.copy(TARGET_FILE, ai_data_file)
        print(f"📂 已创建数据副本: {ai_data_file}")
        
        res_df = pd.DataFrame(results)
        save_df = res_df[['datetime', 'label', 'price']]
        save_df.to_csv(ai_label_file, index=False)
        print(f"💾 已保存标注结果: {ai_label_file}")
    else:
        print("🤷‍♂️ 即使阈值降到了 0.55，还是没找到机会。请检查训练数据是否太少，或模型是否没收敛。")

if __name__ == "__main__":
    main()
