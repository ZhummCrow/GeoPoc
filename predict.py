import os,random,argparse,pickle
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pandas as pd
from data import *
from model import *
from feature_extraction.process_structure import get_pdb_xyz,process_dssp,match_dssp

FOLDS=5
CLASS_DICT={
    "pH":["0-5","5-9","9-14"],
    "salt":["0-0.05%","0.05-4%",">4%"]
}

def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as f:
        sequence_id = None
        sequence = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if sequence_id is not None:
                    sequences[sequence_id] = sequence
                sequence_id = line[1:]
                sequence = ''
            else:
                sequence += line
        if sequence_id is not None:
            sequences[sequence_id] = sequence
    return sequences

def extract_features(prot_dict,args,device):
    ID_list = list(prot_dict.keys())
    seq_list = [prot_dict[i] for i in ID_list]
    
    pdb_exist = 0
    embedding_exist = 0
    dssp_exist = 0
    norm_embedding_exist = 0
    pdb_path = args.feature_path+"pdb/"
    embedding_path = args.feature_path+"embedding/"
    DSSP_path = args.feature_path+"DSSP/"
    for id in ID_list:
        if not os.path.exists(pdb_path + id + '.tensor'):
            pdb_exist = 1
        if not os.path.exists(embedding_path + id + '.pt'):
            embedding_exist = 1
        if not os.path.exists(embedding_path + args.task +"/"+ id + '.tensor'):
            norm_embedding_exist = 1
        if not os.path.exists(DSSP_path + id + '.tensor'):
            dssp_exist = 1
    
    if pdb_exist == 1:
        max_len = max([len(seq) for seq in seq_list])
        chunk_size = 32 if max_len > 1000 else 64
        esmfold_cmd = "python ./feature_extraction/fold.py -i {} -o {} --chunk-size {}".format(args.dataset_path, pdb_path, chunk_size)
        if device=="cpu":
            esmfold_cmd += " --cpu-only"
        else:
            esmfold_cmd = "CUDA_VISIBLE_DEVICES=" + args.gpu + " " + esmfold_cmd
            
        os.system(esmfold_cmd + " | tee ./esmfold_pred.log")
        for ID in tqdm(ID_list):
            with open(pdb_path + ID + ".pdb", "r") as f:
                X = get_pdb_xyz(f.readlines())
            torch.save(torch.tensor(X, dtype = torch.float32), pdb_path + ID + '.tensor')
    if embedding_exist == 1:
        esm_cmd = f"python ./feature_extraction/extract.py esm2_t36_3B_UR50D {args.dataset_path} {embedding_path} --repr_layers 36 --include per_tok"
        if device=="cpu":
            esm_cmd += " --cpu-only"
        else:
            esm_cmd = "CUDA_VISIBLE_DEVICES=" + args.gpu + " " + esm_cmd
        os.system(esm_cmd + " | tee ./esm_emb_pred.log")
    if norm_embedding_exist ==1:
        ESM_MIN_MAX = pickle.load(open("./feature_extraction/ESM_Min_Max.pkl",'rb'))
        MIN = ESM_MIN_MAX[f"{args.task}_Min"]
        MAX = ESM_MIN_MAX[f"{args.task}_Max"]
        for ID in tqdm(ID_list):
            raw_esm = torch.load(f"{embedding_path}{ID}.pt")
            raw_esm = raw_esm['representations'][36].numpy()
            esm_emb = (raw_esm - MIN) / (MAX - MIN)
            torch.save(torch.tensor(esm_emb, dtype = torch.float32), embedding_path + args.task + "/" + ID + '.tensor')
    if dssp_exist == 1:
        for i in tqdm(range(len(ID_list))):
            ID = ID_list[i]
            seq = seq_list[i]
            os.system(f"./feature_extraction/mkdssp -i {pdb_path}{ID}.pdb -o {DSSP_path}{ID}.dssp")
            dssp_seq, dssp_matrix = process_dssp(f"{DSSP_path}{ID}.dssp")
            if dssp_seq != seq:
                dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq)
            torch.save(torch.tensor(np.array(dssp_matrix), dtype = torch.float32), f"{DSSP_path}{ID}.tensor")
            os.system(f"rm {DSSP_path}{ID}.dssp")
        

def pred(args):
    device = torch.device('cuda:'+args.gpu if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path+args.task+"/"
    models = []
    for fold in range(FOLDS):
        state_dict = torch.load(model_path + 'fold%s.ckpt'%fold, device)
        model = GPSoc(task=args.task,device=device).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    
    test_data=read_fasta(args.dataset_path)
    extract_features(test_data,args,device)
    
    test_dataset = ProteinGraphDataset(test_data, range(len(test_data)), args)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, drop_last=False, num_workers=2, prefetch_factor=2)

    test_pred_dict = {} # 导出测试结果
    for data in tqdm(test_dataloader):
        data = data.to(device)
        with torch.no_grad():
            outputs = [model(data.X, data.node_feat, data.edge_index, data.batch) for model in models]
            outputs = torch.stack(outputs,0).mean(0) # 5个模型预测结果求平均
            
        outputs = outputs.detach().cpu().numpy()
        IDs = data.name
        for i, ID in enumerate(IDs):
            test_pred_dict[ID] = outputs[i]

    os.makedirs(args.output_path, exist_ok = True)

    IDs = list(test_pred_dict.keys())
    data={'SeqID':IDs}
    if args.task=="temp":
        data["temperature"]=[test_pred_dict[i] for i in IDs]
    else:
        data[f"raw_{args.task}"]=[test_pred_dict[i] for i in IDs]
        data[f"class_{args.task}"]=[CLASS_DICT[args.task][np.argmax(test_pred_dict[i])] for i in IDs]
    df=pd.DataFrame.from_dict(data)
    df.to_csv(args.output_path+args.task+"_preds.csv",index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dataset_path",type=str,default='./example/test.fasta')
    parser.add_argument("--feature_path",type=str,default='./example/demo/')
    parser.add_argument("--model_path",type=str,default='./model/')
    parser.add_argument("-o", "--output_path",type=str,default='./example/result/')
    parser.add_argument("--task",type=str,default='temp')
    parser.add_argument("--gpu",type=str,default='0')
    
    args = parser.parse_args()
    pred(args)