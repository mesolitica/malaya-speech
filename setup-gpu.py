import setuptools


__packagename__ = 'malaya-speech-gpu'


def readme():
    with open('README-pypi.rst', 'rb') as f:
        return f.read().decode('UTF-8')


with open('requirements.txt') as fopen:
    req = list(filter(None, fopen.read().split('\n')))

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='1.1',
    python_requires='>=3.6.*',
    description='Speech-Toolkit for bahasa Malaysia, powered by Deep Learning Tensorflow. GPU Version',
    long_description=readme(),
    author='huseinzol05',
    author_email='husein.zol05@gmail.com',
    url='https://github.com/huseinzol05/malaya-speech-gpu',
    download_url='https://github.com/huseinzol05/malaya-speech-gpu/archive/master.zip',
    keywords=['nlp', 'bm'],
    install_requires=req + ['tensorflow-gpu>=1.15'],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
