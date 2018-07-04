# MobileNetv2 in PyTorch

An implementation of `MobileNetv2` in PyTorch. 

[Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381) 

I trained on CIFAR-10 dataset as an example to test for Classification . Have not get a ideal result .

# Main Dependencies

   ```
   pytorch 0.4
   python 3.5
 
   ```

# USE

  TEST : when " run_mode" :"test"  in file "config/cifar10_test_exp.json", it can auto classify all the images in "./images"
  ```
  python main.py
  
  ```

  TRAIN : when " run_mode" :"train" "config/cifar10_test_exp.json", and change CIFAR10Data patch in "cifar10data.py"

  ```
  python main.py
  
  ```
  
# Results

 Have not get a ideal result , test on CIFAR10Data.TESTDATA  
 
| ModelSize (M) | Top-1 Accuracy| Top-5 Accuracy|
|---------------|---------------|---------------|
|9.2            |50.3           |93.08          |

