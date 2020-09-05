#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import Birch,AffinityPropagation, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import homogeneity_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler,LabelEncoder, scale


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 7, 4


# In[3]:


def rater(arr2):
    kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
    scaler = StandardScaler()
    scaler.fit(arr2)
    arr2 = scaler.transform(arr2)
    kmeans.fit(arr2)
    y_kmeans = kmeans.predict(arr2)
    ss = silhouette_score(arr2, y_kmeans)
    #print(f"\n\n{ss}\n")
    #print((y_kmeans))
    centers = kmeans.cluster_centers_
    ordrd = [i[0]+i[1] for i in centers]
    unordrd = ordrd.copy()
    ordrd.sort()
    sortd = {unordrd[i]:ordrd.index(unordrd[i]) for i in range(len(ordrd))}
    indxs = []
    for k in range(len(centers)):
        indxs.append(sortd[centers[k][0]+centers[k][1]])
    for f in range(len(y_kmeans)):
        if(y_kmeans[f] == 0):
            y_kmeans[f] = indxs[0]
        elif y_kmeans[f] == 1:
            y_kmeans[f] = indxs[1]
        elif y_kmeans[f] == 2:
            y_kmeans[f] = indxs[2]
        elif y_kmeans[f] == 3:
            y_kmeans[f] = indxs[3]
        elif y_kmeans[f] == 4:
            y_kmeans[f] = indxs[4]


    return y_kmeans


# In[4]:


marg = pd.read_csv('ONT-ALL.csv')
lss = []
# marg.dropna(subset=["Hospitalization_Rate"], inplace=True)
ids = marg['CASE_R'].values.tolist()
# ids = np.asarray(ids)
# scaler = StandardScaler()
# ids = scaler.fit_transform(ids.reshape(-1, 1))
# for i in ids:
#     lss.extend(list(i))
# print(lss)
pops = marg['INSTABILITY_2016'].values.tolist()
arr2 = np.dstack((ids,pops))

hosr = marg['HOSP_R'].values.tolist()
ar3 = np.dstack((hosr,pops))[0]
scaler = StandardScaler()
scaler.fit(ar3)
ar3 = scaler.transform(ar3)
marg['HOSP_STD'] = ar3

dthr = marg['DEATH_R'].values.tolist()
ar4 = np.dstack((dthr,pops))[0]
scaler = StandardScaler()
scaler.fit(ar4)
ar4 = scaler.transform(ar4)
marg['DEATH_STD'] = ar4


arr = np.dstack((ids,pops))[0]
scaler = StandardScaler()
scaler.fit(arr)
arr = scaler.transform(arr)
marg['CASE_STD'] = arr

marg.to_csv('MODDED.csv',index=False)


# In[5]:


arr2 = arr2[0]
instarates = rater(arr2)
#print(instarates)
ids = marg['CASE_R'].values.tolist()
pops = marg['DEPRIVATION_2016'].values.tolist()
arr3 = np.dstack((ids,pops))
arr3 = arr3[0]
deprates = rater(arr3)
#print(deprates)
ids = marg['CASE_R'].values.tolist()
pops = marg['DEPENDENCY_2016'].values.tolist()
arr4 = np.dstack((ids,pops))
arr4 = arr4[0]
depandrates = rater(arr4)
#print(depandrates)
ids = marg['CASE_R'].values.tolist()
pops = marg['ETHNIC-CONC_2016'].values.tolist()
arr5 = np.dstack((ids,pops))
arr5 = arr5[0]
ethrate = rater(arr5)
#print(ethrate)
ids = marg['CASE_R'].values.tolist()
pops = marg['LIM-AT_2016'].values.tolist()
arr6 = np.dstack((ids,pops))
arr6 = arr6[0]
lim16 = rater(arr6)
#print(lim16)
ids = marg['CASE_R'].values.tolist()
pops = marg['HHSIZE_2016'].values.tolist()
arr7 = np.dstack((ids,pops))
arr7 = arr7[0]
hh16 = rater(arr7)
#print(hh16)
ids = marg['CASE_R'].values.tolist()
pops = marg['POPD_2020'].values.tolist()
arr8 = np.dstack((ids,pops))
arr8 = arr8[0]
popd20 = rater(arr8)
#print(popd20)
ids = marg['CASE_R'].values.tolist()
pops = marg['POPR65_2016'].values.tolist()
arr9 = np.dstack((ids,pops))
arr9 = arr9[0]
popr65 = rater(arr9)
#print(popr65)
av = [((instarates[i]+deprates[i]+depandrates[i]+ethrate[i]+lim16[i]+hh16[i]+popd20[i]+popr65[i])/8)+1 for i in range(len(ethrate))]
#print(av)

fig, axs = plt.subplots(4,2,figsize=(15,15))
kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr2)
arr2 = scaler.transform(arr2)
kmeans.fit(arr2)
y_kmeans = kmeans.predict(arr2)
centers = kmeans.cluster_centers_
axs[0,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[0,0].scatter(arr2[:,0],arr2[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[0,0].set(xlabel = "CASE_R")
axs[0,0].set(ylabel = "INSTABILITY_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr3)
arr3 = scaler.transform(arr3)
kmeans.fit(arr3)
y_kmeans = kmeans.predict(arr3)
centers = kmeans.cluster_centers_
axs[1,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[1,0].scatter(arr3[:,0],arr3[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[1,0].set(xlabel = "CASE_RATE")
axs[1,0].set(ylabel = "DEPRIVATION_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr4)
arr4 = scaler.transform(arr4)
kmeans.fit(arr4)
y_kmeans = kmeans.predict(arr4)
centers = kmeans.cluster_centers_
axs[2,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[2,0].scatter(arr4[:,0],arr4[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[2,0].set(xlabel = "CASE_RATE")
axs[2,0].set(ylabel = "DEPENDENCY_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr5)
arr5 = scaler.transform(arr5)
kmeans.fit(arr5)
y_kmeans = kmeans.predict(arr5)
centers = kmeans.cluster_centers_
axs[3,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[3,0].scatter(arr5[:,0],arr5[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[3,0].set(xlabel = "CASE_RATE")
axs[3,0].set(ylabel = "ETHNIC-CONC_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr6)
arr6 = scaler.transform(arr6)
kmeans.fit(arr6)
y_kmeans = kmeans.predict(arr6)
centers = kmeans.cluster_centers_
axs[0,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[0,1].scatter(arr6[:,0],arr6[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[0,1].set(xlabel = "CASE_RATE")
axs[0,1].set(ylabel = "LIM-AT_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr7)
arr7 = scaler.transform(arr7)
kmeans.fit(arr7)
y_kmeans = kmeans.predict(arr7)
centers = kmeans.cluster_centers_
axs[1,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[1,1].scatter(arr7[:,0],arr7[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[1,1].set(xlabel = "CASE_RATE")
axs[1,1].set(ylabel = "HHSIZE_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr8)
arr8 = scaler.transform(arr8)
kmeans.fit(arr8)
y_kmeans = kmeans.predict(arr8)
centers = kmeans.cluster_centers_
axs[2,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[2,1].scatter(arr8[:,0],arr8[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[2,1].set(xlabel = "CASE_RATE")
axs[2,1].set(ylabel = "POPD_2020")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr9)
arr9 = scaler.transform(arr9)
kmeans.fit(arr9)
y_kmeans = kmeans.predict(arr9)
centers = kmeans.cluster_centers_
axs[3,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[3,1].scatter(arr9[:,0],arr9[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[3,1].set(xlabel = "CASE_RATE")
axs[3,1].set(ylabel = "POPR65_2016")

print("")


# In[6]:


k=0
for i in marg['PHU']:
    print(f"{i} rating: {av[k]:.2f}")
    k+=1


# In[7]:


ids = marg['DEATH_R'].values.tolist()
pops = marg['INSTABILITY_2016'].values.tolist()
arr2 = np.dstack((ids,pops))
arr2 = arr2[0]
instarates = rater(arr2)
#print(instarates)
ids = marg['DEATH_R'].values.tolist()
pops = marg['DEPRIVATION_2016'].values.tolist()
arr3 = np.dstack((ids,pops))
arr3 = arr3[0]
deprates = rater(arr3)
#print(deprates)
ids = marg['DEATH_R'].values.tolist()
pops = marg['DEPENDENCY_2016'].values.tolist()
arr4 = np.dstack((ids,pops))
arr4 = arr4[0]
depandrates = rater(arr4)
#print(depandrates)
ids = marg['DEATH_R'].values.tolist()
pops = marg['ETHNIC-CONC_2016'].values.tolist()
arr5 = np.dstack((ids,pops))
arr5 = arr5[0]
ethrate = rater(arr5)
#print(ethrate)
ids = marg['DEATH_R'].values.tolist()
pops = marg['LIM-AT_2016'].values.tolist()
arr6 = np.dstack((ids,pops))
arr6 = arr6[0]
lim16 = rater(arr6)
#print(lim16)
ids = marg['DEATH_R'].values.tolist()
pops = marg['HHSIZE_2016'].values.tolist()
arr7 = np.dstack((ids,pops))
arr7 = arr7[0]
hh16 = rater(arr7)
#print(hh16)
ids = marg['DEATH_R'].values.tolist()
pops = marg['POPD_2020'].values.tolist()
arr8 = np.dstack((ids,pops))
arr8 = arr8[0]
popd20 = rater(arr8)
#print(popd20)
ids = marg['DEATH_R'].values.tolist()
pops = marg['POPR65_2016'].values.tolist()
arr9 = np.dstack((ids,pops))
arr9 = arr9[0]
popr65 = rater(arr9)
#print(popr65)
avdeath = [((instarates[i]+deprates[i]+depandrates[i]+ethrate[i]+lim16[i]+hh16[i]+popd20[i]+popr65[i])/8)+1 for i in range(len(ethrate))]
#print(avdeath)


fig, axs = plt.subplots(4,2,figsize=(15,15))
kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr2)
arr2 = scaler.transform(arr2)
kmeans.fit(arr2)
y_kmeans = kmeans.predict(arr2)
centers = kmeans.cluster_centers_
axs[0,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[0,0].scatter(arr2[:,0],arr2[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[0,0].set(xlabel = "DEATH_RATE")
axs[0,0].set(ylabel = "INSTABILITY_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr3)
arr3 = scaler.transform(arr3)
kmeans.fit(arr3)
y_kmeans = kmeans.predict(arr3)
centers = kmeans.cluster_centers_
axs[1,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[1,0].scatter(arr3[:,0],arr3[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[1,0].set(xlabel = "DEATH_RATE")
axs[1,0].set(ylabel = "DEPRIVATION_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr4)
arr4 = scaler.transform(arr4)
kmeans.fit(arr4)
y_kmeans = kmeans.predict(arr4)
centers = kmeans.cluster_centers_
axs[2,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[2,0].scatter(arr4[:,0],arr4[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[2,0].set(xlabel = "DEATH_RATE")
axs[2,0].set(ylabel = "DEPENDENCY_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr5)
arr5 = scaler.transform(arr5)
kmeans.fit(arr5)
y_kmeans = kmeans.predict(arr5)
centers = kmeans.cluster_centers_
axs[3,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[3,0].scatter(arr5[:,0],arr5[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[3,0].set(xlabel = "DEATH_RATE")
axs[3,0].set(ylabel = "ETHNIC-CONC_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr6)
arr6 = scaler.transform(arr6)
kmeans.fit(arr6)
y_kmeans = kmeans.predict(arr6)
centers = kmeans.cluster_centers_
axs[0,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[0,1].scatter(arr6[:,0],arr6[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[0,1].set(xlabel = "DEATH_RATE")
axs[0,1].set(ylabel = "LIM-AT_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr7)
arr7 = scaler.transform(arr7)
kmeans.fit(arr7)
y_kmeans = kmeans.predict(arr7)
centers = kmeans.cluster_centers_
axs[1,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[1,1].scatter(arr7[:,0],arr7[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[1,1].set(xlabel = "DEATH_RATE")
axs[1,1].set(ylabel = "HHSIZE_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr8)
arr8 = scaler.transform(arr8)
kmeans.fit(arr8)
y_kmeans = kmeans.predict(arr8)
centers = kmeans.cluster_centers_
axs[2,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[2,1].scatter(arr8[:,0],arr8[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[2,1].set(xlabel = "DEATH_RATE")
axs[2,1].set(ylabel = "POPD_2020")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr9)
arr9 = scaler.transform(arr9)
kmeans.fit(arr9)
y_kmeans = kmeans.predict(arr9)
centers = kmeans.cluster_centers_
axs[3,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[3,1].scatter(arr9[:,0],arr9[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[3,1].set(xlabel = "DEATH_RATE")
axs[3,1].set(ylabel = "POPR65_2016")
print("")


# In[8]:


k=0
for i in marg['PHU']:
    print(f"{i} rating: {avdeath[k]:.2f}")
    k+=1


# In[9]:


ids = marg['HOSP_R'].values.tolist()
pops = marg['INSTABILITY_2016'].values.tolist()
arr2 = np.dstack((ids,pops))
arr2 = arr2[0]
instarates = rater(arr2)
#print(instarates)
ids = marg['HOSP_R'].values.tolist()
pops = marg['DEPRIVATION_2016'].values.tolist()
arr3 = np.dstack((ids,pops))
arr3 = arr3[0]
deprates = rater(arr3)
#print(deprates)
ids = marg['HOSP_R'].values.tolist()
pops = marg['DEPENDENCY_2016'].values.tolist()
arr4 = np.dstack((ids,pops))
arr4 = arr4[0]
depandrates = rater(arr4)
#print(depandrates)
ids = marg['HOSP_R'].values.tolist()
pops = marg['ETHNIC-CONC_2016'].values.tolist()
arr5 = np.dstack((ids,pops))
arr5 = arr5[0]
ethrate = rater(arr5)
#print(ethrate)
ids = marg['HOSP_R'].values.tolist()
pops = marg['LIM-AT_2016'].values.tolist()
arr6 = np.dstack((ids,pops))
arr6 = arr6[0]
lim16 = rater(arr6)
#print(lim16)
ids = marg['HOSP_R'].values.tolist()
pops = marg['HHSIZE_2016'].values.tolist()
arr7 = np.dstack((ids,pops))
arr7 = arr7[0]
hh16 = rater(arr7)
#print(hh16)
ids = marg['HOSP_R'].values.tolist()
pops = marg['POPD_2020'].values.tolist()
arr8 = np.dstack((ids,pops))
arr8 = arr8[0]
popd20 = rater(arr8)
#print(popd20)
ids = marg['HOSP_R'].values.tolist()
pops = marg['POPR65_2016'].values.tolist()
arr9 = np.dstack((ids,pops))
arr9 = arr9[0]
popr65 = rater(arr9)
#print(popr65)
avhosp = [((instarates[i]+deprates[i]+depandrates[i]+ethrate[i]+lim16[i]+hh16[i]+popd20[i]+popr65[i])/8)+1 for i in range(len(ethrate))]
#print(avhosp)


fig, axs = plt.subplots(4,2,figsize=(15,15))
kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr2)
arr2 = scaler.transform(arr2)
kmeans.fit(arr2)
y_kmeans = kmeans.predict(arr2)
centers = kmeans.cluster_centers_
axs[0,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[0,0].scatter(arr2[:,0],arr2[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[0,0].set(xlabel = "HOSPITALIZATION_RATE")
axs[0,0].set(ylabel = "INSTABILITY_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr3)
arr3 = scaler.transform(arr3)
kmeans.fit(arr3)
y_kmeans = kmeans.predict(arr3)
centers = kmeans.cluster_centers_
axs[1,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[1,0].scatter(arr3[:,0],arr3[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[1,0].set(xlabel = "HOSPITALIZATION_RATE")
axs[1,0].set(ylabel = "DEPRIVATION_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr4)
arr4 = scaler.transform(arr4)
kmeans.fit(arr4)
y_kmeans = kmeans.predict(arr4)
centers = kmeans.cluster_centers_
axs[2,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[2,0].scatter(arr4[:,0],arr4[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[2,0].set(xlabel = "HOSPITALIZATION_RATE")
axs[2,0].set(ylabel = "DEPENDENCY_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr5)
arr5 = scaler.transform(arr5)
kmeans.fit(arr5)
y_kmeans = kmeans.predict(arr5)
centers = kmeans.cluster_centers_
axs[3,0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[3,0].scatter(arr5[:,0],arr5[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[3,0].set(xlabel = "HOSPITALIZATION_RATE")
axs[3,0].set(ylabel = "ETHNIC-CONC_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr6)
arr6 = scaler.transform(arr6)
kmeans.fit(arr6)
y_kmeans = kmeans.predict(arr6)
centers = kmeans.cluster_centers_
axs[0,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[0,1].scatter(arr6[:,0],arr6[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[0,1].set(xlabel = "HOSPITALIZATION_RATE")
axs[0,1].set(ylabel = "LIM-AT_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr7)
arr7 = scaler.transform(arr7)
kmeans.fit(arr7)
y_kmeans = kmeans.predict(arr7)
centers = kmeans.cluster_centers_
axs[1,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[1,1].scatter(arr7[:,0],arr7[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[1,1].set(xlabel = "HOSPITALIZATION_RATE")
axs[1,1].set(ylabel = "HHSIZE_2016")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr8)
arr8 = scaler.transform(arr8)
kmeans.fit(arr8)
y_kmeans = kmeans.predict(arr8)
centers = kmeans.cluster_centers_
axs[2,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[2,1].scatter(arr8[:,0],arr8[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[2,1].set(xlabel = "HOSPITALIZATION_RATE")
axs[2,1].set(ylabel = "POPD_2020")

kmeans = KMeans(n_clusters=5,init='k-means++', random_state = 42)
scaler = StandardScaler()
scaler.fit(arr9)
arr9 = scaler.transform(arr9)
kmeans.fit(arr9)
y_kmeans = kmeans.predict(arr9)
centers = kmeans.cluster_centers_
axs[3,1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
axs[3,1].scatter(arr9[:,0],arr9[:,1], c=kmeans.labels_.astype(float), cmap='rainbow')
axs[3,1].set(xlabel = "HOSPITALIZATION_RATE")
axs[3,1].set_ylabel("POPR65_2016")
print("")


# In[10]:


k=0
for i in marg['PHU']:
    print(f"{i} rating: {avhosp[k]:.2f}")
    k+=1


# In[11]:


avgall = [(avhosp[i]+avdeath[i]+av[i])/3 for i in range(len(avhosp))]
k=0
for i in marg['PHU']:
    print(f"{i} rating: {avgall[k]:.2f}")
    k+=1

