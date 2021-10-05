Paraview
========

Just some notes on how to interact with Paraview/Python Scripting.

A good resource are the following 2:

1. https://kitware.github.io/paraview-docs/latest/python/quick-start.html
2. Use the Trace Tool within Paraview [Tools > Start Trace]. One can choose 
   to output python code concurrently corresponding to the actions exectued in 
   Paraview or a finished script at the very end [Tools > Stop Trace].
   This has helped me immensely 


Create a separate conda environment for Paraview
++++++++++++++++++++++++++++++++++++++++++++++++

Paraview itself has specific dependencies, so I suggest creating its own
environment in which you install paraview.

.. code:: bash

    conda create -n pv  # follow instructions
    conda activate pv
    conda install paraview
    <run your paraview python script>


**Use scripts and functions located in directory `paraview_tools`**

