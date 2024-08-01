from setuptools import find_packages, setup

# TODO: we should contain some deps


if __name__ == "__main__":
    setup(
        name="H2CEvaluator",
        version="0.1",
        description="Evaluator for human2character algorithms",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="deep learning diffusion pytorch stable diffusion human2character talking head",
        license="Apache 2.0 License",
        author="leo",
        author_email="xingzhening@pjlab.org.cn",
        packages=find_packages("H2CEvaluator"),
        python_requires=">=3.10.0",
        # install_requires=deps,
    )
