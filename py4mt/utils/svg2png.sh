#!/bin/bash

for i in $@; do
  inkscape --without-gui --export-png="$(basename $i .svg).png" $i
done
