from setuptools import setup, find_packages

setup(
    name="esp32-wifi-sensing",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["train_local", "patch_notebooks"],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tensorflow",
    ],
)
