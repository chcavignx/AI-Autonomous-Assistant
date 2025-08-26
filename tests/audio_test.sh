#!/bin/bash
arecord -D plughw:0,0 -f cd -d 5 test.wav
aplay -D plughw:1,0 test.wav