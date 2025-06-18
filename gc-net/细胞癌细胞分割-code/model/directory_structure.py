import os

def print_directory_structure(path, prefix='', is_last=False):
    """递归打印目录结构"""
    # 获取当前路径下的所有文件和文件夹
    items = os.listdir(path)
    items.sort()  # 按名称排序，方便查看
    num_items = len(items)
    
    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_dir = os.path.isdir(item_path)
        prefix_part = '└── ' if i == num_items - 1 else '├── '
        print(f"{prefix}{prefix_part}{item}")
        
        # 递归处理子目录
        if is_dir:
            new_prefix = f"{prefix}{'    ' if i == num_items - 1 else '│   '}"
            print_directory_structure(item_path, new_prefix, i == num_items - 1)

# 示例：打印当前目录结构
print("当前目录结构：")
print_directory_structure(os.getcwd())