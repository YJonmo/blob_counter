### Initial Setup

This is a demo for counting simple blobs in an image using three methods in python and one methods in python+Cpp.

The python methods include:
- Contours (numpy based)
- Laplasian of Gaussian Filters 
- Difference of Gaussian Filters 



The Python+Cpp method is primarry written in Cpp and using Pybind11 it is callable in Python. The Cpp code uses `cv::SimpleBlobDetector` to find the blobs.


### Installation

Clone the repository:

`git clone https://github.com/YJonmo/blob_counter.git`

`cd blob_counter`

Create a conda environment using:

`conda env create -f environment.yml`

And activate it using:

`conda activate blob_counter`

Finally run the code for provided sample images:

`python main.py --image_path ./data/blobs1.tif --min_sigma 20 --max_sigma 60`

and 

`python main.py --image_path ./data/blobs2.tif --thresh_by_area 1 --min_sigma 6 --max_sigma 25`


To call the Cpp code from Python, use following steps to build the Cpp module callable in Python:

`cd pybind`

`mkdir build`

`cd build`

`cmake ..`

`make`

`cd ../`

`python binding.py`


