#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import sys

from setuptools import Command, setup


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Cleaning previous builds...")
            for p in ("build", "dist", *(p for p in os.listdir(setup_dir) if p.endswith(".egg-info"))):
                shutil.rmtree(os.path.join(setup_dir, p), True)
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution...")
        subprocess.run([sys.executable, "-m", "build"], check=True)

        self.status("Uploading the package to PyPI via Twine...")
        dist_dir = os.path.join(setup_dir, "dist")
        subprocess.run(["twine", "upload", *(os.path.join(dist_dir, p) for p in os.listdir(dist_dir))], check=True)

        self.status("Pushing git tags...")
        subprocess.run(["git", "tag", f"v{version}"], check=True)
        subprocess.run(["git", "push", "--tags"], check=True)


setup_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(setup_dir, "README.md"), "r") as f:
    long_description = f.read()

with open(os.path.join(setup_dir, "media_fixer", "__init__.py"), "r") as f:
    version = re.search(r"__version__\s*=\s*['\"](.+)['\"]", f.read()).group(1)


setup(
    author="Halit Şimşek",
    author_email="mail.simsekhalit@gmail.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Multimedia :: Sound/Audio :: Conversion",
    ],
    cmdclass={
        "upload": UploadCommand,
    },
    description="A wrapper around ffmpeg to make it work in a concurrent and memory-buffered fashion.",
    install_requires=["psutil"],
    keywords=["audio", "buffer", "concurrent", "conversion", "decode", "dts", "eac3", "encode", "ffmpeg", "fix",
              "media", "multimedia"],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="media-fixer",
    packages=["media_fixer"],
    python_requires=">=3.8",
    url="https://github.com/simsekhalit/media-fixer",
    version=version,
)
