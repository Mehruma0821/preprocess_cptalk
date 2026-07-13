from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='preprocess_cptalk',
    version='0.1.0',
    description='Text preprocessing utilities for NLP tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='CPTalk',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'preprocess_cptalk': ['data/*.json'],
    },
)