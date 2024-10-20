from setuptools import setup, find_packages

# requirements.txt 파일을 읽어서 install_requires에 넣어주는 함수
def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="instruction ip-adapter",
    version="0.1.0",
    description="A instruction based ip-adapter package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Noah",
    author_email="noah@aimmo.co.kr",
    url="https://github.com/bluewhalekr/ip_adapter_instruction.git",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),  # requirements.txt 파일을 참조
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
