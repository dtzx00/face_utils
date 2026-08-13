import io
import os

from setuptools import setup


def readme():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), "r", encoding="utf-8") as handle:
        return handle.read()


setup(
    name="face_utils",
    version="1.0.0+study2",
    description="Portrait segmentation, Face++ alignment, and grayscale preprocessing for Study 2.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    license="BSD-2-Clause",
    packages=["face_utils"],
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31",
        "numpy>=1.24",
        "opencv-python-headless>=4.8",
        "pillow>=10.0",
        "pillow-avif-plugin>=1.5",
        "python-dotenv>=1.0",
        "mediapipe>=0.10.14,<2",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
