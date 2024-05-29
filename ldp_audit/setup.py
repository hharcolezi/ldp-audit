from setuptools import setup, find_packages

setup(
    name='ldp_audit',
    version='0.1.0',
    description='A package for auditing Local Differential Privacy protocols',
    author='Heber H. Arcolezi',
    author_email='hh.arcolezi@gmail.com',
    url='https://github.com/hharcolezi/ldp_audit',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'multi-freq-ldpy==0.2.5',
        'numba==0.59.1',
        'numpy==1.26.4',
        'pandas',
        'pure-ldp==1.1.2',
        'psutil',
        'ray==2.20.0',
        'scipy',
        'statsmodels',
        'tqdm',        
        'xxhash',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
