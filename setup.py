from setuptools import setup, find_packages

setup(
    name="enhanced_fever_medication_recommendation_system",
    version="1.0.0",
    description="An end-to-end ML pipeline for fever medication recommendation (Paracetamol vs Ibuprofen).",
    author="Oscar Owino",
    packages=find_packages(include=["src", "scripts", "src.*", "scripts.*"]),
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "fever-pipeline=main:run_all",   # allows running pipeline via CLI
        ],
    },
    python_requires=">=3.8",
)
