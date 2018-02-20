from setuptools import setup, find_packages

setup(
    name='tf_custom',
    description='Custom modules for rapid development.',
    packages=find_packages(exclude=('data')),
    python_requires='>=3.5',
    install_requires=[
        'tensorflow',
        'matplotlib',
        'numpy',
        'pandas',
        'sklearn'
    ]
)
