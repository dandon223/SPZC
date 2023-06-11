# %%
import pandas as pd


df = pd.read_csv("attack_dataset.csv") # attack dataset
bonafide = pd.read_csv('bonafide_dataset_20191121.csv.gz') # bonafide traffic from mawilab
bonafide = pd.concat([bonafide, pd.read_csv('bonafide_dataset_20201110.csv.gz')])
bonafide = pd.concat([bonafide, pd.read_csv('bonafide_dataset_20201129.csv')])
print(df.shape, bonafide.shape)

bonafide['label'] = "bonafide"

if (df.columns == bonafide.columns).all():
    examples_malicious = df.shape[0]
    examples_legitim = bonafide.shape[0]
    total = examples_malicious+examples_legitim
    print('Total examples of {0} with {1:0.2f} of attack and {2:0.2f} bonafide packets'.format(total, examples_malicious/total, examples_legitim/total))

fields = ['eth.type', 'ip.id', 'ip.flags', 'ip.checksum', 'ip.dsfield', 'tcp.flags', 'tcp.checksum']

for field in fields:
    df[field] = df[field].apply(lambda x: int(str(x), 16))

bonafide = bonafide.fillna(0)
for field in fields:
    bonafide[field] = bonafide[field].apply(lambda x: int(str(x), 16))

full_data = pd.concat([bonafide, df])

wrong_proto = full_data[full_data['ip.proto'] != 6]['label'].value_counts().values
full_data = full_data[full_data['ip.proto'] == 6]
print("It was found and removed", wrong_proto, "packets.")

full_data.drop(columns=['frame_info.time', 'frame_info.encap_type', 'frame_info.time_epoch', 'frame_info.number',
                        'frame_info.len', 'frame_info.cap_len', 'eth.type', 'ip.flags', 'ip.src', 'ip.dst',
                        'ip.version', 'ip.proto', 'tcp.flags'], axis=1, inplace=True)

full_data.info()
full_data.describe()
# check features with zero variance (not useful for learning)

# remove columns with zero variance
full_data.drop(columns=['ip.hdr_len', 'ip.tos', 'ip.flags.rb',
                        'ip.flags.mf', 'ip.frag_offset'], axis=1, inplace=True)

full_data['label'].value_counts()
full_data.label[full_data.label == "bonafide"] = 0  # convert bonafide label to 0
full_data.label[full_data.label != 0] = 1  # convert attack labels to 1
full_data['label'].value_counts()

full_data.drop(columns=["ip.checksum", "ip.ttl", "tcp.checksum", "tcp.dstport", "tcp.seq", "tcp.srcport",
                        "tcp.ack", "tcp.options.mss_val"], axis=1, inplace=True)

full_data.to_csv("MachineLearningCVE/ABTRAP_Dataset.csv", index=False)
