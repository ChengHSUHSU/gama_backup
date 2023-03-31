import setuptools

REQUIRED_PACKAGES = [
    'xgboost==1.4.2', 'pandas==1.2.4', 'numpy==1.19.5', 'torch==1.8.1', 'scikit-learn==0.24.2', 'tqdm==4.60.0'
]

setuptools.setup(
    name="planetrecsys",
    version="0.1.0",
    python_requires=">=3.8",
    install_requires=REQUIRED_PACKAGES,
    author="SeanChen",
    author_email="xiangchen@gamania.com",
    description="",

    packages=setuptools.find_packages(exclude=["src", "utils"])
)
