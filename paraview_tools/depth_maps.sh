#!/bin/bash

dir=$1


source ~/anaconda3/etc/profile.d/conda.sh


depths=(0 5 10 15 20 30 40 50 60 70)
params=(vpv vsv rho)

# depths=(5)
# params=(vpv)

for param in "${params[@]}";
do
    for depth in "${depths[@]}";
    do

        # Get parameter
        file=${dir}/*${param}.vtu

        # Create csv of depth slice
        conda activate pv
        echo ${file}, ${param}, ${depth}
        python ./depthslice2csv.py -f ${file} -o ${dir}/${param}.csv -p ${param} -d ${depth}
        conda deactivate

        # Create PDF
        conda activate lwsspy
        echo ${file}, ${param}, ${param}
        plot_csv_depth_slice -f ${dir}/${param}.csv -o ${dir}/${param} -l ${param}
        conda deactivate
    done
done