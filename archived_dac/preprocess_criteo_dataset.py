from data_utils import getCriteoAdData

getCriteoAdData(
    datafile="/home/chenggu2/datasets/train.txt",
    o_filename='kaggleAdDisplayChallenge_processed.npz',
    max_ind_range=-1,
    sub_sample_rate=0.0,
    days=7,
    data_split='train',
    randomize='total',
    criteo_kaggle=True,
    memory_map=False
)