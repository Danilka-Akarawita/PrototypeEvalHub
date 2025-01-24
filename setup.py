from setuptools import setup, find_packages

setup(
    name='text_evaluator_prototype',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'transformers',
        'torch',
        'nltk',
        'mauve',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'text_evaluator_app=text_evaluator.streamlit_app:main',
        ],
    },
    author='danilka-akarawaita',
    author_email='danilka.20210514@iit.ac.lk',
    description='A library for evaluating text using various metrics.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Danilka-Akarawita/PrototypeEvalHub',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)