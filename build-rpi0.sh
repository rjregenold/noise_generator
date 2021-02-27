#!/bin/bash

docker build -t rjregenold/cross:arm-rpi-4.9.3-linux-gnueabihf docker &&\
  cross clean &&\
  cross build --target arm-unknown-linux-gnueabihf --release
