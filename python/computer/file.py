import os
import zlib
import argparse

def decompress_zlib_to_jpg(input_dir, overwrite=True):
    """
    将目录内所有 .zlib 文件解压为 .jpg
    :param input_dir: 输入目录（默认当前目录）
    :param overwrite: 是否覆盖已存在的 .jpg 文件
    """
    # 遍历目录内所有文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".zlib"):
                zlib_path = os.path.join(root, file)
                # 输出路径：同目录下，后缀改为 .jpg
                jpg_path = os.path.join(root, os.path.splitext(file)[0] + ".jpg")
                
                # 跳过已存在的 .jpg（若不覆盖）
                if not overwrite and os.path.exists(jpg_path):
                    print(f"跳过：{jpg_path} 已存在")
                    continue
                
                try:
                    # 读取 zlib 压缩文件
                    with open(zlib_path, "rb") as f_in:
                        compressed_data = f_in.read()
                    
                    # 解压 zlib 数据
                    decompressed_data = zlib.decompress(compressed_data)
                    
                    # 写入 .jpg 文件
                    with open(jpg_path, "wb") as f_out:
                        f_out.write(decompressed_data)
                    
                    print(f"成功：{zlib_path} → {jpg_path}")
                except Exception as e:
                    print(f"失败：{zlib_path} → 错误：{str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量解压 .zlib 文件为 .jpg")
    parser.add_argument("-d", "--dir", default=".", help="输入目录（默认当前目录）")
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite", help="不覆盖已存在的 .jpg 文件")
    args = parser.parse_args()
    
    decompress_zlib_to_jpg(args.dir, args.overwrite)