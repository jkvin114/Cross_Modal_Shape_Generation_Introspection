"""
DO NOT RUN OR IMPORT.
THIS IS SEPARATE SCRIPT FROM THE PROJECT AND WONT BE ABLE TO RUN.
THIS IS USED FOR AUTOMATICALLY COLORIING MESH FACES/SECTIONS USING K-MEANS ALGORITHM

"""

import numpy as np  
import json
from pyclustering.cluster.kmeans import kmeans

import open3d as o3d;
from pyclustering.utils.metric import type_metric, distance_metric
import sys
K=10
K2=2
import colorsys
GROUP_COLORS=np.array([colorsys.hsv_to_rgb(c/(K),0.8,0.8) for c in range(K)])
print(GROUP_COLORS)# 0.2+c/(K*K2) * 0.6,0.2+c/(K*K2) * 0.6

lam=1  # difference btw surcface normals
gam=0.1 # bias term 
beta=0.2 # plainarity term

myfunc = lambda point1, point2: beta * np.dot(point1[3:6],point2[:3])  ** 2\
        + gam * (1- np.dot(point1[3:6],point2[3:6])) ** 2\
        + lam * np.mean(point1[3:6]-point2[3:6])**2 

def distance(p1, p2):
    return np.sum((p1 - p2)**2)


def initialize(data, k):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points having shape (200, 2)
        k - number of clusters
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])
    # plot(data, np.array(centroids))
  
    ## compute remaining k - 1 centroids
    for c_id in range(k - 1):
         
        ## initialize a list to store distances of data
        ## points from nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize
             
            ## compute distance of 'point' from each of the previously
            ## selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
             
        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        # plot(data, np.array(centroids))
    return centroids
  

def KMeans(sample,metric,k):
    max=np.max(sample)


    metric2 = distance_metric(type_metric.USER_DEFINED, func=metric)
    
    initial_centers = np.random.normal(loc=0,scale=max,size=(k,6))
    initial_centers= initialize(sample,k)

    kinstance = kmeans(sample, initial_centers, metric=metric2,itermax =10,tolerance=0.01)

    kinstance.process()

    clstr = kinstance.predict(sample)
    # print(clstr)
    return np.array(clstr) 

def show_mesh(shapeid):
    mesh=np.load(f"chairs/chairs/chairs_meshes/{shapeid}.polygons.npy")
    colors=np.load(f"output/{shapeid}.color.npy")
    numtri=mesh.shape[0]
    meshpoly=mesh
    v,indices=np.unique(meshpoly.reshape(numtri*3, 3), axis=0,return_inverse=True)
    tri=(indices).reshape(numtri, 3)
    mesht = o3d.t.geometry.TriangleMesh()

    mesht.vertex.positions  = o3d.core.Tensor(v)
    mesht.triangle.indices  = o3d.core.Tensor(tri)
    mesht.triangle.colors=o3d.core.Tensor(colors)
    o3d.visualization.draw([mesht])   

def process_mesh(shapeid):
    print(shapeid)
    mesh=np.load(f"chairs/chairs/chairs_meshes/{shapeid}.polygons.npy")
    numtri=mesh.shape[0]
    meshpoly=mesh
    v,indices=np.unique(meshpoly.reshape(numtri*3, 3), axis=0,return_inverse=True)
    tri=(indices).reshape(numtri, 3)
    centriods=np.zeros((tri.shape[0],3))
    for i,t in enumerate(tri):
        coords=v[t]
        centriods[i]=np.mean(coords,axis=1)

    # print(shapeid)
    mesh = o3d.geometry.TriangleMesh()
    # Use mesh.vertex to access the vertices' attributes    
    mesh.vertices = o3d.utility.Vector3dVector(v)
    # Use mesh.triangle to access the triangles' attributes    
    mesh.triangles = o3d.utility.Vector3iVector(tri)
    mesh.compute_triangle_normals(normalized=True)
    normals_normalized=mesh.triangle_normals
    concatedface_norm=np.concatenate([centriods,normals_normalized],axis=1)


    groups=KMeans(concatedface_norm,myfunc,K)

    # =========================================  this part will be ignored
    for i in range(K,0):
        sub_indices=np.argwhere(groups==i)
        if sub_indices.shape[0]==0: continue

        facegroup=concatedface_norm[sub_indices].reshape((sub_indices.shape[0],6))
        norm = np.mean(facegroup[3:6],axis=0)
        # print(facegroup)
        def comp(p1,p2):
            
            return np.mean(((p2[:3]-p1[:3])**2)*(1-p2[3:6])) + 0.01 * (np.dot(p1[3:6],p2[3:6])) ** 2
            return (np.linalg.norm((p2[3:6])-np.linalg.norm(p1[3:6])))**2
        subgroups=KMeans(facegroup,comp,K2)
        # print(subgroups*K2 +i)
        groups[sub_indices] = subgroups.reshape((-1,1)) +i*K2
        print(groups)

    # =========================================


    # print(groups)
    colors=GROUP_COLORS[groups]

    # print(shapeid)
    mesht = o3d.t.geometry.TriangleMesh()

    mesht.vertex.positions  = o3d.core.Tensor(v)
    mesht.triangle.indices  = o3d.core.Tensor(tri)
    mesht.triangle.colors=o3d.core.Tensor(colors)
    # o3d.visualization.draw([mesht])   
    np.save(f"output/{shapeid}.color.npy",colors)
    # np.save(f"output/{shapeid}.faces.npy",mesht.triangle.indices)
    # np.save(f"output/{shapeid}.vertices.npy",mesht.vertex.positions)
    # o3d.io.write_triangle_mesh("mesh.obj",mesht,print_progress=True)


def processall():
    images=[]
    with open("sv2_chairs_test_small.json") as f:
        images=json.load(f)['ShapeNetV2']['03001627']
        # print(images)
    with open("sv2_chairs_train_small.json") as f:
        images+=json.load(f)['ShapeNetV2']['03001627'][:40]
        # print(images)

    images.append("1013f70851210a618f2e765c4a8ed3d")
    images.append("308b76aac4b518a43eb67d9fb75cc878")
    # 1a6f615e8b1b5ae4dbbc9440457e303e
    print(images[0])


    for i,im in enumerate(images):
        print(f"{i}/{len(images)} Processing............................................................")
        process_mesh(im)


"""

@inproceedings{inproceedings,
author = {Garland, Michael and Willmott, Andrew and Heckbert, Paul},
year = {2001},
month = {03},
pages = {49-58},
title = {Hierarchical Face Clustering on Polygonal Meshes},
journal = {Proceedings of the Symposium on Interactive 3D Graphics},
doi = {10.1145/364338.364345}
}"""



if __name__=="__main__":
    # processall()
    show_mesh("11c7675a3dbc0d32f7287e3d21227e43")