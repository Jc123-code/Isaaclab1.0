import torch
import os
from collections import OrderedDict
import json

def print_dict_recursive(d, indent=0, max_depth=5):
    """
    递归打印字典内容
    """
    if indent > max_depth:
        print("  " * indent + "... (超过最大深度)")
        return
    
    for key, value in d.items():
        prefix = "  " * indent
        
        if isinstance(value, dict):
            print(f"{prefix}[{key}] (dict, {len(value)} 个键)")
            print_dict_recursive(value, indent + 1, max_depth)
        elif isinstance(value, list):
            print(f"{prefix}[{key}] (list, {len(value)} 个元素)")
            if len(value) > 0 and len(value) <= 10:
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        print(f"{prefix}  [{i}]:")
                        print_dict_recursive(item, indent + 2, max_depth)
                    else:
                        print(f"{prefix}  [{i}]: {item}")
            elif len(value) > 10:
                print(f"{prefix}  前3个: {value[:3]}")
                print(f"{prefix}  ... (共{len(value)}个)")
        elif isinstance(value, torch.Tensor):
            print(f"{prefix}[{key}] Tensor: shape={list(value.shape)}, dtype={value.dtype}")
        elif isinstance(value, str) and len(value) > 100:
            print(f"{prefix}[{key}] (str): {value[:100]}...")
        else:
            print(f"{prefix}[{key}]: {value}")

def analyze_pth_file_detailed(pth_path):
    """
    详细分析PyTorch .pth文件的内容
    """
    print("=" * 80)
    print(f"分析文件: {pth_path}")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(pth_path):
        print(f"错误: 文件不存在 {pth_path}")
        return
    
    # 显示文件大小
    file_size = os.path.getsize(pth_path) / (1024 * 1024)  # MB
    print(f"\n文件大小: {file_size:.2f} MB")
    
    # 加载文件
    try:
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
        print("✓ 文件加载成功\n")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return
    
    # 显示顶层结构
    print("=" * 80)
    print("顶层结构:")
    print("=" * 80)
    if isinstance(checkpoint, dict):
        print(f"顶层键: {list(checkpoint.keys())}\n")
    
    # 详细显示每个部分
    print("=" * 80)
    print("详细内容:")
    print("=" * 80)
    
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"\n{'='*80}")
            print(f"[{key}]")
            print(f"{'='*80}")
            
            value = checkpoint[key]
            
            if key == 'model':
                # 特殊处理模型参数
                print(f"类型: {type(value).__name__}")
                if isinstance(value, dict):
                    print(f"包含 {len(value)} 个键: {list(value.keys())}\n")
                    for sub_key, sub_value in value.items():
                        print(f"\n  [{sub_key}]")
                        if isinstance(sub_value, dict):
                            analyze_state_dict(sub_value, indent="    ")
                        else:
                            print(f"    类型: {type(sub_value).__name__}")
            
            elif key == 'config':
                # 配置信息已经很详细了，可以跳过或简化
                print("类型: str (配置JSON)")
                print("(配置内容较长，已在上面显示)")
            
            elif key in ['env_metadata', 'shape_metadata']:
                # 详细显示这些元数据
                print(f"类型: {type(value).__name__}")
                if isinstance(value, dict):
                    print(f"包含 {len(value)} 个键\n")
                    print_dict_recursive(value, indent=1)
            
            else:
                # 其他键的通用处理
                print(f"类型: {type(value).__name__}")
                if isinstance(value, dict):
                    print(f"包含 {len(value)} 个键\n")
                    print_dict_recursive(value, indent=1)
                elif isinstance(value, (str, int, float, bool, type(None))):
                    print(f"值: {value}")
                else:
                    print(f"内容: {str(value)[:200]}")

def analyze_state_dict(state_dict, indent=""):
    """
    分析state_dict的详细信息
    """
    print(f"{indent}参数统计:")
    print(f"{indent}" + "-" * 60)
    
    total_params = 0
    total_size = 0
    
    print(f"{indent}总共 {len(state_dict)} 个参数\n")
    
    # 显示每个参数
    for i, (name, param) in enumerate(state_dict.items(), 1):
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            size_mb = num_params * param.element_size() / (1024 * 1024)
            total_params += num_params
            total_size += size_mb
            
            print(f"{indent}{i}. {name}")
            print(f"{indent}   形状: {list(param.shape)}")
            print(f"{indent}   数据类型: {param.dtype}")
            print(f"{indent}   参数量: {num_params:,}")
            print(f"{indent}   大小: {size_mb:.4f} MB")
    
    print(f"\n{indent}" + "=" * 60)
    print(f"{indent}总参数量: {total_params:,}")
    print(f"{indent}总大小: {total_size:.2f} MB")
    print(f"{indent}" + "=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python analyze_pth.py <model.pth>")
        sys.exit(1)
    
    pth_path = sys.argv[1]
    analyze_pth_file_detailed(pth_path)
