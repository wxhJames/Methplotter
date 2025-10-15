import subprocess
import sys
packages = ['setuptools', 'wheel','twine']
for package in packages:
    try:
        # 尝试导入包，检查是否已安装
        __import__(package)
        print(f"{package} 已经安装")
    except ImportError:
        print(f"{package} 未安装，正在安装...")
        # 使用 pip 安装包
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} 安装成功")

import setuptools  # 导入setuptools打包工具
import wheel
import twine

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Methplotter",  # 用自己的名替换其中的YOUR_USERNAME_
    version="1.5",  # 包版本号，便于维护版本,保证每次发布都是版本都是唯一的
    author="wxhJames",  # 作者，可以写自己的姓名
    author_email="1937557685@qq.com",  # 作者联系方式，可写自己的邮箱地址
    description="Methplotter",  # 包的简述
    long_description=long_description,  # 包的详细介绍，一般在README.md文件内
    long_description_content_type="text/markdown",
    url="https://github.com/wxhJames/Methplotter",  # 自己项目地址，比如github的项目地址
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # 对python的最低版本要求
)
