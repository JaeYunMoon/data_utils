import os,glob
from collections import Counter
import cv2
import matplotlib
import matplotlib.pyplot as plt
from abc import *
from pathlib import Path

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class Confirm(metaclass=ABCMeta):
    def __init__(self,pth) -> None:
        # self.im = cv2.imread(img,cv2.COLOR_BGR2RGB)

        if not os.path.exists(pth):
            os.makedirs(pth)

        self.save_path = pth
        self.font = cv2.FONT_HERSHEY_PLAIN
    @abstractmethod
    def class_num_graph(self):
        pass 

    
    def bbox_confirm(self):
        pass 

class yolo_dataset_confirm(Confirm):
    def __init__(self,data_txts,pth,labels_dict:dict) :
        self.labels = labels_dict
        super().__init__(pth)
        self.data_txts = data_txts
        self.gt_parent = Path(data_txts[0]).parent
    def class_num_graph(self,task = "",xticks_rt = 0,xticks_fs = 10):
        count_ls = [] 
        for dt in self.data_txts:
            suf = Path(dt).suffix
            assert suf == ".txt", f"The data format is wrong. need .txt but it is now file suffix {suf}"
            f = open(dt,"r")
            lines = f.readlines() 
            for line in lines:
                c = line.split(" ")[0]
                count_ls.append(c)
        label_count = Counter(count_ls)
        sort_class_count = sorted(label_count.items())  # list
        keys = [] 
        values = [] 
        for i,v in sort_class_count:
            keys.append(int(i))
            values.append(v)

        matplotlib.use('svg')  # faster
        #plt.rcParams["figure.figsize"] = [8, 6]
        plt.rcParams["figure.autolayout"] = True # tight_layout
        bars = plt.bar(keys,values)
        plt.xticks(rotation=xticks_rt,fontsize = xticks_fs)
        plt.xticks(keys,self.labels.values())
        
        for bar in bars:
            height = bar.get_height()
            plt.text(x=bar.get_x() + bar.get_width()/2,y=height+0.5,s = "{}".format(height),ha = "center")

        plt.title(f"{task} Label Count(image_num : {len(self.data_txts)})")
        sv = os.path.join(self.save_path,f"{task}class_num_graph.png")
        plt.savefig(sv,dpi=200)
        matplotlib.use('Agg')
        plt.close()
        print(sv)

    def bbox_confirm(self,data_imgs,colors):
        sp = os.path.join(self.save_path,"image_confirm/")
        print(os.path.join(self.save_path,"image_confirm/"))
        print(sp)
        if not os.path.exists(sp):
                os.makedirs(sp)
        for img in data_imgs:
            im = cv2.imread(img,cv2.COLOR_BGR2RGB)
            y,x,ch = im.shape
            img_pth = Path(img)
            img_name = img_pth.name
            img_suf = img_pth.suffix
            gt = os.path.join(self.gt_parent,str(img_name).replace(img_suf,".txt"))
            gt = Path(gt)

            if str(gt) in [i.replace("/",os.sep) for i in self.data_txts]:
                f = open(gt,"r")
                lines = f.readlines()
                for line in lines:
                    c,xc,yc,w,h = line.split(" ")
                
                    xc,yc,w,h = float(xc)*x,float(yc)*y,float(w)*x,float(h)*y
                    xmin = int((xc - (w//2)))
                    xmax = int((xc + (w//2)))
                    ymin = int((yc - (h//2)))
                    ymax = int((yc + (h//2)))
                    print(c,xc,yc,w,h)
                    c = int(c)
                    print(xmin,xmax,ymin,ymax,c)
                    #im = cv2.rectangle(im,(xmin,ymin),(xmax,ymax),([x / 255 for x in colors(c)]),3)
                    im = cv2.rectangle(im,(xmin,ymin),(xmax,ymax),colors[c],3)
                    im = cv2.putText(im,self.labels[c],(int(xmin-5),int(ymin-5)),self.font,2,colors[c],1,cv2.LINE_AA)
            
            sv = os.path.join(sp,img_name)
            
            cv2.imwrite(sv,im)
        return os.path.join(self.save_path,f"/image_confirm/")
    # 색상이 주어지지 않을 때, 랜덤으로 색상을 지정해 줄 수 있도록 수정 하고자함 10.11 
    # 현재 yolov5 코드를 참고하여 색상을 넣고 있음 




    
