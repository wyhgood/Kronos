# test_gpu.py
import torch

def test_pytorch_gpu():
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")

        # 测试GPU计算
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("GPU计算测试：成功！")
    else:
        print("警告：CUDA不可用，将使用CPU模式")

if __name__ == "__main__":
    test_pytorch_gpu()
