# xgcm

[![Build Status](https://travis-ci.org/xgcm/xgcm.svg?branch=master)](https://travis-ci.org/xgcm/xgcm)
[![codecov.io](https://codecov.io/github/xgcm/xgcm/coverage.svg?branch=master)](https://codecov.io/github/xgcm/xgcm?branch=master)

xgcm is a python package for analyzing general circulation model (GCM) output data.
xgcm is built on top of [xray](http://github.com/xray/xray).

xgcm is motivated by the fact that our models are getting bigger and
[bigger](http://maps.actualscience.net/MITgcm_llc_maps/llc_4320/),
and we are in desperate need of some scalable analysis tools.

The main functionality xgcm adds on top of xray is an more complex representation
of the [grids](see https://en.wikipedia.org/wiki/Arakawa_grids) commonly used in
numerical modeling and an implementation of differential and integral operators
(e.g. gradients) in the specific ways appropriate to these grids.
