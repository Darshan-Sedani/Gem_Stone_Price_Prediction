from setuptools import find_packages,setup
from typing import List

HYPEN_E_GOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPEN_E_GOT in requirements:
            requirements.remove(HYPEN_E_GOT)
        
    return requirements

setup(
name='gem_price_prediction',
version='0.0.1',
author="Darshan Sedani",
package=find_packages(),
install_requires=get_requirements('requirements.txt')
)