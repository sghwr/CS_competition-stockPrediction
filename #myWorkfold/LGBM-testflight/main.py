"""
LightGBM回归模型训练与预测 - 主启动脚本
适用于Colab远程服务器环境
"""

import os
import sys
import subprocess
import platform
import argparse


def check_dependencies():
    """检查并安装必要的依赖包"""
    print("检查依赖包...")

    required_packages = [
        "lightgbm",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "scipy",
        "tqdm",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except (ImportError, OSError):
            missing_packages.append(package)
            print(f"  ✗ {package} (未安装或加载失败)")

    if missing_packages:
        print(f"\n发现 {len(missing_packages)} 个未安装的包")
        response = input("是否自动安装? (y/n): ").strip().lower()
        if response == "y":
            print("开始安装依赖包...")
            for package in missing_packages:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package]
                    )
                    print(f"  ✓ 已安装 {package}")
                except subprocess.CalledProcessError:
                    print(f"  ✗ 安装 {package} 失败")
            print("依赖包安装完成!")
        else:
            print("请手动安装缺失的包:")
            print(f"pip install {' '.join(missing_packages)}")
            return False

    return True


def setup_directories():
    """创建必要的目录结构"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    directories = [
        os.path.join(script_dir, "model"),
        os.path.join(script_dir, "output"),
        os.path.join(script_dir, "logs"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"目录已确保: {directory}")

    return script_dir


def check_data_file(script_dir):
    """检查数据文件是否存在"""
    # 尝试多种可能的数据路径
    possible_paths = [
        os.path.join(
            os.path.dirname(script_dir), "data", "train.csv"
        ),  # 项目根目录/data/train.csv
        os.path.join(script_dir, "..", "data", "train.csv"),  # 上级目录/data/train.csv
        os.path.join(script_dir, "data", "train.csv"),  # 当前目录/data/train.csv
        "/content/THU-BDC2026/data/train.csv",  # Colab常见路径
        "./data/train.csv",  # 相对路径
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到数据文件: {path}")
            return path

    print("错误: 未找到数据文件 train.csv")
    print("请将数据文件放在以下位置之一:")
    for path in possible_paths:
        print(f"  - {path}")

    custom_path = input("\n请输入数据文件完整路径 (或按Enter退出): ").strip()
    if custom_path and os.path.exists(custom_path):
        return custom_path

    return None


def print_banner():
    """打印程序横幅"""
    banner = """
    ╔══════════════════════════════════════════════════╗
    ║      LightGBM股票收益率预测系统 (Colab版)        ║
    ║                                                  ║
    ║  基于回归模型直接预测未来5日收益率               ║
    ║  替代排序学习，解决任务错配问题                  ║
    ╚══════════════════════════════════════════════════╝
    """
    print(banner)


def run_training(data_path):
    """运行训练脚本
    Args:
        data_path: 数据文件路径
    """
    print("\n" + "=" * 60)
    print("开始训练LightGBM回归模型")
    print("=" * 60)
    print(f"使用数据文件: {data_path}")

    try:
        # 导入训练模块
        from train_lgbm import main as train_main

        train_main(data_path)
        return True
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_prediction(data_path):
    """运行预测脚本
    Args:
        data_path: 数据文件路径
    """
    print("\n" + "=" * 60)
    print("开始运行收益率预测")
    print("=" * 60)
    print(f"使用数据文件: {data_path}")

    try:
        # 导入预测模块
        from predict_lgbm import main as predict_main

        predict_main(data_path)
        return True
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return False


def interactive_menu(data_path):
    """交互式菜单"""
    while True:
        print("\n" + "═" * 50)
        print("请选择操作:")
        print("  1. 训练模型")
        print("  2. 运行预测")
        print("  3. 训练并预测 (完整流程)")
        print("  4. 检查环境和配置")
        print("  5. 查看输出结果")
        print("  6. 退出程序")
        print("═" * 50)

        choice = input("请输入选项 (1-6): ").strip()

        if choice == "1":
            run_training(data_path)
        elif choice == "2":
            run_prediction(data_path)
        elif choice == "3":
            if run_training(data_path):
                print("\n训练完成，开始预测...")
                run_prediction(data_path)
        elif choice == "4":
            check_environment()
        elif choice == "5":
            show_results()
        elif choice == "6":
            print("感谢使用，再见!")
            break
        else:
            print("无效选项，请重新输入")


def check_environment():
    """检查运行环境"""
    print("\n" + "=" * 60)
    print("环境检查")
    print("=" * 60)

    print(f"Python版本: {platform.python_version()}")
    print(f"操作系统: {platform.system()} {platform.release()}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"脚本目录: {script_dir}")

    # 检查GPU
    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: 不可用 (使用CPU)")
    except:
        print("GPU: 未检测 (未安装PyTorch)")

    # 检查关键目录
    model_dir = os.path.join(script_dir, "model")
    output_dir = os.path.join(script_dir, "output")

    print(
        f"模型目录: {model_dir} - {'存在' if os.path.exists(model_dir) else '不存在'}"
    )
    print(
        f"输出目录: {output_dir} - {'存在' if os.path.exists(output_dir) else '不存在'}"
    )

    # 检查模型文件
    model_files = ["lgbm_model.pkl", "feature_columns.pkl"]
    for file in model_files:
        file_path = os.path.join(model_dir, file)
        exists = os.path.exists(file_path)
        print(f"模型文件 {file}: {'存在' if exists else '不存在'}")

    print("=" * 60)


def show_results():
    """显示输出结果"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")

    print("\n" + "=" * 60)
    print("输出结果")
    print("=" * 60)

    if not os.path.exists(output_dir):
        print("输出目录不存在")
        return

    result_files = os.listdir(output_dir)
    if not result_files:
        print("输出目录为空")
        return

    for file in result_files:
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        print(f"  {file}: {file_size:,} bytes")

        if file == "result.csv" and file_size > 0:
            try:
                import pandas as pd

                df = pd.read_csv(file_path)
                print(f"    内容预览 (Top5股票):")
                print(df.to_string(index=False))
            except:
                print("    无法读取文件内容")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LightGBM股票收益率预测系统")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--predict", action="store_true", help="运行预测")
    parser.add_argument("--full", action="store_true", help="训练并预测（完整流程）")
    parser.add_argument("--data-path", type=str, help="指定数据文件路径")
    parser.add_argument("--check", action="store_true", help="检查环境和配置")
    parser.add_argument("--results", action="store_true", help="查看输出结果")
    args = parser.parse_args()

    # 如果没有任何参数，显示横幅并进入交互模式
    interactive_mode = not (
        args.train or args.predict or args.full or args.check or args.results
    )

    if not interactive_mode:
        # 非交互模式：不显示横幅，直接执行
        pass
    else:
        print_banner()

    # 检查依赖
    if not check_dependencies():
        print("依赖检查失败，请手动安装所需包")
        return

    # 设置目录
    script_dir = setup_directories()

    # 处理不需要数据文件的操作
    if args.check:
        check_environment()
        return
    if args.results:
        show_results()
        return

    # 需要数据文件的操作：训练、预测、完整流程、交互模式
    if args.data_path:
        data_path = args.data_path
        if not os.path.exists(data_path):
            print(f"错误: 指定的数据文件不存在: {data_path}")
            return
    else:
        data_path = check_data_file(script_dir)
        if not data_path:
            print("数据文件检查失败，程序退出")
            return

    print(f"\n使用数据文件: {data_path}")

    # 设置环境变量，供其他模块使用
    os.environ["DATA_PATH"] = data_path

    # 非交互模式执行
    if args.train:
        run_training(data_path)
        return
    if args.predict:
        run_prediction(data_path)
        return
    if args.full:
        if run_training(data_path):
            print("\n训练完成，开始预测...")
            run_prediction(data_path)
        return

    # 交互模式
    interactive_menu(data_path)


if __name__ == "__main__":
    main()
