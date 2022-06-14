# BIRA-NET-Bilinear-Attention-Net-for-Diabetic-Retinopathy-Grading
This is a PyTorch implementation of BIRA-NET paper. You can find it's official implementation also but available repository does not have whole code. They have given only RA part of the model. There is one discrpancy in my code. I could not match the exact model structure after the part of outer product which is given in the last paragraph of section 3.3. There are two code files. One model has my modified CNN backbone to extract features (it contains dilated convolutions, you can edit it according to your use case). Other model has Resnet-18 as backbone. 

<!-- ![alt text](https://github.com/sourabh-patil/BIRA-NET-Bilinear-Attention-Net-for-Diabetic-Retinopathy-Grading/blob/master/biranet.png?raw=true) -->
![biranet](https://user-images.githubusercontent.com/53788836/173560243-d574e604-695f-4bcf-9787-3cea2c86d8ec.png)
