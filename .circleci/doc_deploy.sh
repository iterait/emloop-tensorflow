#!/usr/bin/env bash

# prepare the env for building docs
apt-get install -y python3-sphinx graphviz locales language-pack-en
pip3 install -r requirements-docs.txt
locale-gen --purge
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
git config --global user.email "cxflow@cognexa.com"
git config --global user.name "CircleCI"

# build the docs
cd docs
sphinx-build . build

# push the docs to the gh-pages branch
cd ..
git stash
git checkout gh-pages
find . -maxdepth 1 -not -path '*/\.*' ! -name 'docs' -exec rm -rf {} \;
cp -r docs/build/* .
rm -rf docs
git add --all
git commit -m "Docs update from $CIRCLE_BRANCH : $CIRCLE_SHA1"
git push
