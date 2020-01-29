#! /bin/bash
find . -name \*.edi -exec sed -i "s/PAX/GEO/g" {} \;
