from setuptools import setup, find_packages

__version__ = "0.2.0"

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="image_dedupe",
    version=__version__,
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=requirements,
    author="Rafael Padilla",
    author_email="eng.rafaelpadilla@gmail.com",
    description="A tool for finding and managing duplicate images",
    url="https://github.com/rafaelpadilla/image_dedupe",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
