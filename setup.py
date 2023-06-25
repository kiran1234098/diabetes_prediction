from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT='-e .' # Triggers setup.py file... we will not count is as library it in requirements... 

def get_requirements(file_path:str)->List[str]:
    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='Diabetes',
    version='0.0.1',
    author='Kiran_zapate',
    author_email='kiranzapate@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)