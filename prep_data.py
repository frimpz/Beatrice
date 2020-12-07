import pickle

import pandas as pd
import numpy as np
import email
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import networkx as nx

def get_file_info(text ):
    dats=[]
    name=[]
    mtype=[]
    mnum =[]
    for i in range(len(text)):
        dats.append(text['file'].iloc[i].split('/'))
        name.append(dats[i][0])
        mtype.append(dats[i][1])
        if len(dats[i])==2:
            mnum.append(0)
        else:
            mnum.append(dats[i][2])
    df_info =pd.DataFrame({'Name': name,
     'Type': mtype,
     'Num': mnum
    })
    return df_info


def get_messages(field, messages):
    cname = []
    for mes in messages:
        received = email.message_from_string(mes)
        cname.append(received.get(field))
    return cname


def message_body(messages):
    cname = []
    for message in messages:
        e = email.message_from_string(message)
        cname.append(e.get_payload())
    return cname

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


emails = pd.read_csv('graphs\emails.csv').head(50000)
cols = emails.columns

df_info = get_file_info(emails)

message = emails.loc[100]["message"]
e = email.message_from_string(message)
mdf = pd.DataFrame()

mdf["date"] = get_messages("Date", emails["message"])
mdf["From"] = get_messages("From", emails["message"])
mdf["To"] = get_messages("To", emails["message"])
mdf["Subject"] = get_messages("Subject", emails["message"])
mdf["Mime-Version"] = get_messages("Mime-Version", emails["message"])
mdf["Content-Type"] = get_messages("Content-Type", emails["message"])
mdf["X-From"] = get_messages("X-From", emails["message"])
mdf["X-To"] = get_messages("X-To", emails["message"])
mdf["X-cc"] = get_messages("X-cc", emails["message"])
mdf["X-bcc"] = get_messages("X-bcc", emails["message"])
mdf["X-Folder"] = get_messages("X-Folder", emails["message"])
mdf["X-Origin"] = get_messages("X-Origin", emails["message"])
mdf["X-FileName"] = get_messages("X-FIlename", emails["message"])
mdf["message_body"] = message_body(emails["message"])

Combined_df = pd.concat([df_info, mdf], axis=1)
data = Combined_df.dropna(subset=['To'])

Sinfo= pd.DataFrame()
Sinfo['From'] = data['From']
Sinfo['To'] = data['To']

G_symmetric = nx.Graph()
G=nx.from_pandas_edgelist(Sinfo, 'From', 'To')
print(nx.info(G))
print(G.adj)
print(type(G.adj))


A = nx.adjacency_matrix(G)
A = A.todense()
print(A)
print(type(A))

with open('test1.pkl', 'wb') as f:
     pickle.dump(A, f)


with open('test1.pkl', 'rb') as f:
    x = pickle.load(f)
    print(x.shape)

print(check_symmetric(x))