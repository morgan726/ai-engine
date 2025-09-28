from openpi.shared import download

# 指定自定义下载路径
custom_download_dir = "/path/to/your/custom/directory"
checkpoint_dir = download.maybe_download(
    "s3://openpi-assets/checkpoints/pi0_libero",
    save_dir=custom_download_dir  # 传入自定义路径参数
)