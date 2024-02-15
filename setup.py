from setuptools import setup, find_packages

setup(
    name='tegridy-tools',
    version='2.14.2024',    
    description='All tegridy-tools as python modules',
    url='https://github.com/asigalov61/tegridy-tools',
    keywords="tegridy tools midi artificial intelligence machine learning deep learning",
    author='Aleksandr Sigalov',
    author_email='No public email atm',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=['tqdm',
                      'numpy',
                      'matplotlib',                 
                      ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 2',
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"
    ],
)
