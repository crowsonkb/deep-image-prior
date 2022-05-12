from setuptools import setup, find_packages

setup(
    name='deep-image-prior',
    packages=find_packages(exclude=[]),
    version='0.0.1',
    license='Apache',
    description='Deep Image Prior w/ deformable convolutions',
    keywords=[
        "deep image prior",
        'artificial intelligence',
        'deep learning',
        'text to image'
    ],
    install_requires=[
        "jupyter",
        "numpy",
        "pyyaml",
        "mkl",
        "setuptools",
        "cmake",
        "cffi",
        "matplotlib",
        "scikit-image",
        "torch>=1.8.2",
        "torchvision",
    ],
)
