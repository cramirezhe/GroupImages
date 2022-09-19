# Created by Carlos Ramirez at 16/09/2022
import platform
from distutils.core import setup
from group_images import __author__, __email__, __version__

id_os = platform.system()

requirements = [
        'matplotlib~=3.6.0',
        'numpy~=1.23.0',
        'scikit-learn~=1.1.2',
        'opencv-python~=4.6.0.66'
]

if id_os == 'Darwin':
    requirements.append('tensorflow-macos==2.9.2')
    requirements.append('tensorflow-metal==0.5.1')
elif id_os == 'Linux':
    requirements.append('tensorflow==2.9.2')
else:
    raise OSError(f"{id_os} is not sopported")

setup(
    name='GroupImages',
    version=__version__,
    author=__author__,
    author_email=__email__,
    url='https://github.com/cramirezhe/GroupImages',
    install_requires=requirements,
    python_requires='>=3.8.9',
    packages=[
        'group_images.feature_extractor',
        'group_images.separate'
    ],
    entry_points={
        'console_scripts': ['cluster_images = group_images.main:terminal_exec']
    }
)