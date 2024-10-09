from setuptools import setup, find_packages



def get_requirements(filepath):
    with open(filepath) as f:
        return f.read().splitlines()



setup(
    name="MLProject",
    version="1.0",
    packages=find_packages(),
    author="ukasha",
    author_email="ukashaatif123@gmail.com",
    install_requires=get_requirements("requirements.txt"), 
)