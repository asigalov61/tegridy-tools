from setuptools import setup

setup(
    name='tegridy-tools',
    version='1.0.1',    
    description='All tegridy-tools as python modules',
    url='https://github.com/asigalov61/tegridy-tools',
    keywords="tegridy tools midi artificial intelligence machine learning deep learning",
    author='Aleksandr Sigalov',
    author_email='asigalov61@hotmail.com',
    license='Apache 2.0',
    packages=['tegridy_tools'],
    install_requires=['tqdm',
                      'numpy',                     
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
        "Programming Language :: Python :: 3.9"
    ],
)