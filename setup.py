from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]
        # Remove '-e .' if present
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='naveen',
    author_email='naveenkumarchapala123@gmail.com',
    packages=find_packages(),
    python_requires='>=3.11,<3.14',
    install_requires=get_requirements('requirements.txt')
)
