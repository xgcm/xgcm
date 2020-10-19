# How to release a new version of xgcm (for maintainers only)
The process of releasing at this point is very easy. 

We need only two things: A doc PR and and making a release on github.

1. Make sure that all the new features/bugfixes etc are appropriately documented in `doc/whats-new.rst`, add the date to the current release and make an empty (unreleased) entry for the next minor release as a PR.
2. Navigate to the 'tags' symbol on the repos main page, click on 'Releases' and on 'Draft new release' on the right.
  - Add the version number and a short description and save the release.
  
From here the github actions take over and package things for [Pypi](https://pypi.org/project/xgcm/).
The conda-forge package will be triggered by the Pypi release and you will have to approve a PR in [xgcm-feedstock](https://github.com/conda-forge/xgcm-feedstock). This takes a while, usually a few hours to a day.

Thats it!
