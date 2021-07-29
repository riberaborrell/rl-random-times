import setuptools

import os

with open("VERSION", "r") as f:
    VERSION = f.read().strip("\n")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rl-for-gym',
    version=VERSION,
    description='Reinforcement Learning algorithms for gym environments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/eborrell/rl-for-gym",
    project_urls={
        "Bug Tracker": "https://github.com/eborrell/rl-for-gym/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    #license='GNU General Public License V3',
    author='Enric Ribera Borrell',
    author_email='ribera.borrell@me.com',
    keywords=[],
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'gym',
        'numpy',
        'matplotlib',
        'torch',
    ],
    extras_require={
        'test': ['pytest'],
    },
)
