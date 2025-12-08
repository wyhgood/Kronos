import torch
from model import Kronos

MODEL_PATH = "NeoQuasar/Kronos-base"

def main():
    print(f"🔍 正在解剖模型: {MODEL_PATH} ...")
    try:
        model = Kronos.from_pretrained(MODEL_PATH)
        
        # 深入模型内部找 Embedding 层
        # 根据报错堆栈: s1_emb = self.emb_s1(s1_ids)
        # 我们直接找 emb_s1
        if hasattr(model, 'embedding'):
            emb_layer = model.embedding
            if hasattr(emb_layer, 'emb_s1'):
                weight = emb_layer.emb_s1.weight
                print(f"\n✅ 找到 S1 Embedding 层!")
                print(f"📏 真实物理形状: {weight.shape}")
                print(f"🛑 最大允许 Token ID: {weight.shape[0] - 1}")
                
                # 顺便检查 S2
                if hasattr(emb_layer, 'emb_s2'):
                    w2 = emb_layer.emb_s2.weight
                    print(f"📏 S2 真实物理形状: {w2.shape}")
            else:
                print("❌ 没找到 emb_s1，模型结构可能不同。")
        else:
            print("❌ 没找到 embedding 模块。")
            
    except Exception as e:
        print(f"❌ 加载失败: {e}")

if __name__ == "__main__":
    main()
