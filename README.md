# isprs_seg
ISPRS Vaihingen and Potsdam Datasets Segmentation

datasets web: 

https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/

https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/

tools/cut_data.py
#generate the image slices for training

train.py
#train your model

test_inference.py
#test the source large size image. pixel resolution range is (0~20000). 
#TO FUTURE-> (20000~+∞)

tools/metrics.py
#calculate the metrics same as the isprs official website.