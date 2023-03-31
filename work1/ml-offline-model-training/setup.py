from setuptools import setup, find_packages


install_requires = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    install_requires = [line.strip() for line in f.readlines()]

setup(
    name='BDDSLIB',
    version='0.0.3',
    author='Gamania Data Center',
    description='Testing installation of Package',
    long_description='',
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    packages=find_packages(where='.', include=('bdds_recommendation*')),
)
