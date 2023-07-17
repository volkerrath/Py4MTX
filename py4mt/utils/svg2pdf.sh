#!/bin/bash

for i in $@; do
  inkscape --without-gui --export-pdf="$(basename $i .svg).pdf" $i
done
