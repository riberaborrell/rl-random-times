from setuptools import setup, find_packages

import os


with open('VERSION', 'r') as f:
    VERSION = f.read().strip('\n')

_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(_dir, 'README.md'), 'r') as f:
    README = f.read()


setup(
    name='RL_for_gym',
    version=VERSION,
    description='Reinforcement Learning algorithms for gym environments',
    long_description_content_type='text/markdown',
    long_description=README,
    classifiers=[],
    url='https://github.com/eborrell/RL_for_gym',
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
