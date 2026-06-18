#!/bin/bash


iter=14		#best model, constant_start_19
#iter2=9999	#9thdecile/1stdecile from constant_start ensemble

cut=EW

# Plot settings

width=-15/15
size=0.05

Hrange=-15/15
#Vrange=-15/7
#AreaSize=15/11.

Vrange=-41/7
AreaSize=15/24.

ah=5
fh=5
av=5
fv=5

# Plot settings
#Hrange=-4.79/4.79
#Vrange=-5/3
#AreaSize=7/5.84
#ah=1
#fh=1
#av=1
#fv=1
#cut=SVO-05



# Prepare color palette
gmt makecpt -Cjet -Z -I -T0/3/0.2 > Res.cpt

# --- draw color section ---
#gmt set ANOT_FONT Helvetica
gmt set FONT_ANNOT_PRIMARY 12p,Helvetica,black
gmt set FONT_ANNOT_SECONDARY 12p,Helvetica,black
#gmt set HEADER_FONT Helvetica
gmt set FONT_HEADING 12p,Helvetica,black
#gmt set LABEL_FONT Helvetica
gmt set FONT_LABEL 12p,Helvetica,black
#gmt set ANOT_FONT_SIZE 20pt HEADER_FONT_SIZE 14pt LABEL_FONT_SIZE 20pt
gmt set MAP_FRAME_WIDTH 0.1c
gmt set MAP_TICK_LENGTH_PRIMARY 0.1c 
gmt set MAP_ANNOT_OFFSET_PRIMARY 0.2c
gmt set COLOR_NAN 255/255/255
gmt set MAP_FRAME_TYPE PLAIN
gmt set MAP_FRAME_PEN thin
gmt set MAP_LABEL_OFFSET 5p

# Start GMT session
gmt begin Model_profile_${cut}_alpha1 pdf
gmt subplot begin 2x2 -Fs$AreaSize -M0c/0c -R$Hrange/$Vrange -JX$AreaSize



#plot resistivity

#replace value of iter in the param file:
awk -v iter="$iter" 'NR == 2 {print iter; next} {print}' param_V_${cut}.txt > tmp_param.dat
mv tmp_param.dat param_V_P${cut}.dat

makeCutawayForGMT.x param_V_P${cut}.dat

in_file="resistivity_GMT_iter${iter}.dat"
ps_data="tmp.txt"
value_file="value.txt"

grep Z $in_file | awk '{z=$3; if (z < 0) z=0; else if (z > 4) z=4; print z}' > "$value_file"
awk '{if ($1 == ">") print $0; else printf "%15.6e %15.6e\n", -$1, -$2 }' $in_file > "$ps_data"

#grep Z $in_file | awk '{z=$3; if (z < 0) z=0; else if (z > 4) z=4; print z}' > "$value_file"
#awk '{if ($1 == ">") print $0; else printf "%15.6e %15.6e\n", $1, -$2 }' $in_file > "$ps_data"

Boption=a${ah}f${fh}:"y(km)":/a${av}f${fv}:"z(km)":WSne

gmt subplot set 0

	gmt plot "$ps_data" -CRes.cpt -G+z -Z"$value_file" -Bxa5f5+l"distance along profile (km)" -Bya5f5+l"z (km)" -BWSne -L -V0 --FONT_LABEL=14p,Helvetica,black

	
	awk -F',' 'NR>1 {print $2, -$3}' P${cut}_station_names.csv | \
	gmt plot -Sc0.1c -Gblack -W0.25p
	
	awk -F',' 'NR>1 {printf "%.6f %.6f %s\n", $2, -$3-0.1, $1}' P${cut}_station_names.csv | \
	gmt pstext -F+f8p,Helvetica,black+jTL+a90 -N

	awk -F',' 'NR>1 {print -$1, -$2}' ${cut}_EQ.csv | \
	gmt plot -Sc0.1c -Gwhite -W0.25p

	

gmt subplot set 1
# Horizontal colorbar below all plots, centered at x=0, y=-1.5c
#gmt psscale -CRes.cpt -Dx0c/4.7c+w"$g_width"c/0.4c+h -Bxa1f1+l"Resistivity [log(@~W@~m)]" -N
#gmt psscale -CRes.cpt -Dx0c/4.7c+w"$g_width"c/0.4c+h -Bxa1f1+l"Resistivity [log(@~W@~m)]" -N
gmt psscale -CRes.cpt -Dx0c/4.7c+w3c/0.4c+h -Bxa1f1+l"[log(@~W@~m)]" -N --FONT_LABEL=14p,Helvetica,black --FONT_ANNOT_PRIMARY=14p,Helvetica,black

gmt subplot end

gmt end
echo "All plots completed."
