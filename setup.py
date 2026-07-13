import io, os
from setuptools import setup


def readme():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()


install_requirements = [
    "requests",
    "scipy",
    "numpy",
    "pandas",
    "opencv-python-headless",
    "scikit-learn",
    "matplotlib",
    "python-dotenv",
    "statsmodels",
]

setup(
    name='face_utils',
    version='0.3.0',
    description='Facial-feature extraction, image preprocessing, and statistics utilities for facial-image research.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/dtzx00/Facial_Features',
    author='Dawei Wang',
    author_email='david.wang@kellogg.northwestern.edu',
    license='BSD 2-clause',
    packages=['face_utils'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
    install_requires=install_requirements,
    extras_require={
        # Deep-learning extras: only needed for load_custom_vgg / VGGFace features.
        "deep": ["tensorflow>=2.0"],
    },
)
