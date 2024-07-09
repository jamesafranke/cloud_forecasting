import os, torch, numpy as np, random
from PIL import Image
import demo_util

config = demo_util.get_config("/share/data/2pals/jim/data/1d-tokenizer/configs/titok_l32.yaml")

tokenizer = demo_util.get_titok_tokenizer(config)
device = "cuda"
tokenizer = tokenizer.to(device)

fl = np.loadtxt('/share/data/2pals/jim/data/geostat/truecolor.txt', dtype='str')
random.shuffle(fl)
rez = 256

for file in fl:
    out = torch.empty((16,32))
    date = file.split('_')[-1].split('.')[0]
    path = f'/share/data/2pals/jim/data/geostat/tokenized/{date}.pt'
    if os.path.exists(path) == False:
        x = torch.from_numpy(np.array(Image.open(file)).astype(np.float32)).permute(2, 0, 1) / 255.0
        temp = x[:,1000:1000+rez*4,1300:1300+rez*4].unfold(1, rez, rez).unfold(2, rez, rez).contiguous().view(3, -1, rez, rez)
        
        for i in range(temp.shape[1]): 
            out[i,:] = tokenizer.encode(temp[:,i,:,:].unsqueeze(0).to(device))[1]["min_encoding_indices"].to("cpu", dtype=torch.uint8)
        torch.save(out.flatten(), path)