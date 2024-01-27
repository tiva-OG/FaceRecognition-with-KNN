from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_packages(file_path: str) -> List[str]:
    """
    this function returns a list of the required packages

    :param file_path: path to file containing requirements
    :return: list of requirements
    """

    with open(file_path) as requirements_file:
        requirements = requirements_file.readlines()
        requirements = [req.replace("/n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


if __name__ == '__main__':
    setup(
        name='FaceRecognition-with-KNN',
        version='1.0.0',
        author='tiva',
        packages=find_packages(),
        install_requires=get_packages('requirements.txt')
    )
