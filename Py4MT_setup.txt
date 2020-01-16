The mtpy environment in Anaconda can be generated from the 
mtpy.yaml file (exported from Volker's working environment) 
by entering: 

conda env create -f mtpy.yml

If this works fine, carry on with:


conda activate mtpy


Then get your git repository (using the develop branch at your own risk) with


git clone https://github.com/MTgeophysics/mtpy.git whatever-directory-you-want
cd whatever-directory-you-want
python setup.py install

That should do the trick. You can then test calling

mtpy_tests.sh

Enjoy, 
Volker

