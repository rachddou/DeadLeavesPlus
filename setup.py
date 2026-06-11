from setuptools import setup, find_packages

setup(
    name="dead_leaves_generation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'pillow',
        'scikit-image',
        'scipy',
        'matplotlib',
        'omegaconf',
        'shapely',
        'rasterio',
    ],
    python_requires='>=3.7',
    description="Dead leaves texture generation",
    author="Raphael",
)
