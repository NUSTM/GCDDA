import argparse
def build_args(parser):
    """Build arguments."""
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="rest")
    return parser.parse_args()
args = build_args(argparse.ArgumentParser())
co_guess_data=[]

data1,domain=args.data_dir.split('#')
s,t=domain.split('-')
model1_guess_data=[]
model2_guess_data=[]
with open(data1+domain+'/pre.txt','r') as fout1:
    for line1 in fout1:
        text1,label1,label2=line1.strip().split('***')
        text1=text1.split()
        label1=label1.split()
        label2=label2.split()
        texts=[]
        labels=[]
        for i in range(len(text1)):
            if texts==[] and text1[i].startswith('##'):
                continue
            
            if not text1[i].startswith('##'):
                texts.append(text1[i])
                labels.append(label1[i])
            else:
                # print(text1[i],texts)
                texts[-1]=texts[-1]+text1[i][2:]
                
        if label1==label2:
            model1_guess_data.append(' '.join(texts)+'####'+' '.join(labels))

      
data_dir, domain= args.output_dir.split('#')
import os

os.makedirs(data_dir, exist_ok=True) 
with open(data_dir+domain+'.txt','w')as fin:
    fin.write('\n'.join(model1_guess_data))

