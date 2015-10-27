from setuptools import setup, find_packages

classifiers = [
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
]

setup (
    name='WikiCat',
    version='0.0.1',
    author='Zachary Jablons',
    url='https://github.com/zmjjmz/WikiCat',
    py_modules = ['WikiCat'],
    install_requires = ['beautifulsoup4', 'scikit-learn', 'numpy', 'requests'],
    classifiers = classifiers
)
