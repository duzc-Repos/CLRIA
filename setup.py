from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
README = (this_directory / "README.md").read_text()

setup(
    name='clria',
    version='1.0.5',
    author='DU Zongchang',
    license='MIT',
    url='https://github.com/duzc-Repos/CLRIA',
    
    description='Package to decipher LRI-mediated brain network communication',
    long_description=README,
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    
    include_package_data=True,
    package_data = {
        '':['*.txt', '*.xlsx', '*.tsv', '*.csv']
    },
    packages=find_packages(exclude=['CLRIA_reproducibility', 'tests*']),
    
    install_requires=[
        'numpy',  'pandas', 'tensorly',
        'scipy', 'statsmodels',
        'scikit_learn'
        'nibabel', 'netneurotools',
        'POT'
        'plotly'
    ],
    python_requires='>=3.8',
    
)
