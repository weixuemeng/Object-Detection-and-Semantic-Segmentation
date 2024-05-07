# Object-Detection-and-Semantic-Segmentation
## Task
The task is to do object detection and semantic segmentation on the MNIST Double Digits RGB (MNISTDD-RGB) dataset. This dataset contains RGB images of size 64×64 where each has two digits from 0 to 9 placed randomly inside it.

## Dataset
The MNISTDD-RGB dataset is divided into 3 subsets: train, validation and test containing 55K, 5K and 10K samples respectively. A sample consists of: <br />
### Image: 
A 64×64×3 image that has been vectorized to a 12288-dimensional vector. <br />
### Labels: 
A 2-dimensional vector that has two numbers in the range [0, 9] which are the two digits in the image. <br /> 
+ These two numbers are always in ascending order. For example, if digits 7 and 5 are in an image, then this two-vector will be [5, 7] and not [7, 5]. </br>
### Bounding boxes: 
A 2×4 matrix which contains two bounding boxes that mark the locations of the two digits in the image. The first row contains the location of the first digit in labels and the second row contain the location of the second one.
Each row of the matrix has 4 numbers which represent [𝑦_𝑚𝑖𝑛,𝑥_𝑚𝑖𝑛,𝑦_𝑚𝑎𝑥,𝑥_𝑚𝑎𝑥] in this exact order, where:
- 𝑦_𝑚𝑖n = row of the top left corner
- 𝑥_𝑚𝑖𝑛 = column of the top left corner
- 𝑦_𝑚𝑎𝑥 = row of the bottom right corner
- 𝑥_𝑚𝑎𝑥 = column of the bottom right corner </br>
It is always the case that 𝑥_𝑚𝑎𝑥 – 𝑥_𝑚𝑖𝑛 = 𝑦_𝑚𝑎𝑥 – 𝑦_𝑚𝑖𝑛 = 28. This means that each bounding box has a size of 28×28 no matter how large or small the digit inside that box is.
### Segmentation Mask: 
A 64×64 image with pixel values in the range [0, 10], where 10 represents the background.
