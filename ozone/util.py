import torch

def cudaify(x):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            cuda = torch.device('cuda:2')
        else:
            cuda = torch.device('cuda:0')
        return x.cuda(cuda)
    else: 
        return x

if torch.cuda.is_available():
    print("using gpu")
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    print("using cpu")
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
