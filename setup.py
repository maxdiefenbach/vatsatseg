import io
from setuptools import setup


with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    README = readme_file.read()

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name='vatsatseg',
    version='0.0.1',
    description=('segment Visceral and Subcutaneous Adipose Tissue (VAT, SAT)'
                 'in water-fat MRI images'),
    long_description=README,
    author='Maximilian N. Diefenbach',
    author_email='maximilian.diefenbach@tum.de',
    url='https://github.com/maxdiefenbach/vatsatseg',
    packages=['vatsatseg'],
    data_files=[('vatsatseg', ['vatsatseg/config.ini',
                               'vatsatseg/label_desc.txt'])],
    install_requires=REQUIREMENTS,
    entry_points={
        'console_scripts': ['vatsatseg = vatsatseg:cli'],
    },
    tests_require=['pytest'],
    include_package_data=True,
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Version Control :: Git',
        'Topic :: Text Editors :: Emacs'
    ],
    keywords=('mri, radiology, water-fat imaging, segmentation'),
)
