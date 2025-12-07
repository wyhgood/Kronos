# test_kronos_installation.py
import sys
import torch
import pandas as pd
import numpy as np

def test_basic_imports():
    """测试基础包导入"""
    print("测试基础包导入...")
    try:
        import matplotlib
        import tqdm
        import einops
        from huggingface_hub import hf_hub_download
        print("✅ 基础包导入成功")
        return True
    except ImportError as e:
        print(f"❌ 基础包导入失败: {e}")
        return False

def test_torch_functionality():
    """测试PyTorch功能"""
    print("测试PyTorch功能...")
    try:
        # 测试基本张量操作
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        z = torch.matmul(x, y)

        # 测试GPU（如果可用）
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            print("✅ PyTorch GPU功能正常")
        else:
            print("⚠️  使用CPU模式")

        print("✅ PyTorch功能测试通过")
        return True
    except Exception as e:
        print(f"❌ PyTorch功能测试失败: {e}")
        return False

def test_pandas_functionality():
    """测试pandas功能"""
    print("测试pandas功能...")
    try:
        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = {
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }
        df = pd.DataFrame(data, index=dates)

        print(f"✅ 创建测试数据成功，形状: {df.shape}")
        print("✅ pandas功能测试通过")
        return True
    except Exception as e:
        print(f"❌ pandas功能测试失败: {e}")
        return False

def test_kronos_import():
    """测试Kronos模块导入"""
    print("测试Kronos模块导入...")
    try:
        # 添加项目路径
        sys.path.append('.')
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ Kronos模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ Kronos模块导入失败: {e}")
        print("提示：请确保您在Kronos项目根目录下运行此脚本")
        return False

def main():
    """主测试函数"""
    print("开始Kronos安装验证测试...\n")

    tests = [
        test_basic_imports,
        test_torch_functionality,
        test_pandas_functionality,
        test_kronos_import
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print("-" * 50)

    print(f"\n测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！Kronos环境配置成功！")
        return True
    else:
        print("⚠️  部分测试失败，请检查上述错误信息并修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


