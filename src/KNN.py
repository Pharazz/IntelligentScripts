class KNNCluster:
  #Epochs Ran?
  #Cluster Type?
  def __init__(self, name = "Cluster1", points=[]):
    self.name = name
    self.points = points
  def rename(self, name):
    self.name = name
  def setPoints(self, points):
    self.points = points
  def removeFromCluster(self, point):
    self.points.remove(point)
  def addToCluster(self, point):
    self.points.append(point)
  def emptyCluster(self):
    self.points.clear()
  def getPoints(self):
    return np.array(self.points)
  def chkPointinCluster(self, point):
    return np.any(np.all(point == self.points, axis=1)) #If all of the columns match in any of the points
  def __str__(self):
    return "---\n KNN-Cluster: " + self.name + "\nContains: {} points\n".format(len(self.points)) + str(self.points) +"\n---"

#Get distance matrix of all points, if data size N matrix will be NxN /
#Add first point in cluster /
#For all i points in N,
# check distance matrix between points 0:point[i] and find minimum
# Find index of minimum euc distance
# If minimum exceeds threshold, find cluster of indexed point
#   Add point i to cluster
# If minimum does not exceed threshold, make new cluster j add point i to cluster j

def KNNAlg(points=[], threshold = 4):
  Clusters = [None] * len(points)
  #Distance from point i to j in [i][j]
  distMat = getDistMat(points)
  # print(distMat)
  Clusters[0] = copy.deepcopy(KNNCluster("Cluster0",[points[0]]))
  NumOfClusters = 1
  for i in range(1,len(points)):
    #Check Distances
    # print(i)
    # print(distMat[i])
    if (i == 1):
      minDist = distMat[i][0]
    else:
      minDist = np.min(distMat[i][:i])
    # print("MinDist: {}".format(minDist))
    if (minDist > threshold):
      Clusters[NumOfClusters] = copy.deepcopy(KNNCluster("Cluster"+str(NumOfClusters), [points[i]]))
      NumOfClusters+=1
      #make new cluster
    else: #find cluster and add to it
      #Get index of Minimum distance
      pointIDX = np.squeeze(np.where(distMat[i][0:i] == minDist)[0][0])
      #Get index of cluster from index of point
      for j in range(NumOfClusters):
        if Clusters[j].chkPointinCluster(points[pointIDX]):
          ClusterIDX = j
          # print("ClusterIDX {}".format(j))
          break
      Clusters[ClusterIDX].addToCluster(points[i])
    
  filtered_items = filter(lambda item: item is not None, Clusters)
  new_Clusters = list(filtered_items)
  return new_Clusters

def plotClustersKNN(ClusterList, Title = "K-NN Clustering", Subtitle = "Epoch = 1", Xlabel = "X", Ylabel = "Y"):
  # colorArray = ['red', 'blue', 'green', 'orange', 'pink',]
  # colorsList = ["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"]
  fig1, ax1 = plt.subplots()
  ax1.grid(visible = True, which = 'both')
  ax1.set_title(Subtitle)
  fig1.suptitle(Title)
  ax1.set_xlabel(Xlabel)
  ax1.set_ylabel(Ylabel)
  # ax1.set_xlim([0,10])
  for i in range(len(ClusterList)):
    # ax1.scatter(ClusterList[i].getPoints()[:,0],ClusterList[i].getPoints()[:,1], label=('Cluster' + str(i)))
    ax1.scatter(ClusterList[i].getPoints()[:,0],ClusterList[i].getPoints()[:,1], label=(ClusterList[i].name))
  ax1.legend(loc = "best")
  return
# Test1 = KMeansAlg([[0,0], [0,1], [0,2]])

def getGuesses(ClusterList, points):
  GuessList = []
  for i in range(len(points)):
    for j in range(len(ClusterList)):
      if points[i] in ClusterList[j].getPoints():
        GuessList.append(j)
        break
  return np.array(GuessList)

def PlotBinlblData(data, label,range, Title = "Raw Data", Subtitle = "Heart Disease", Xlabel = "SBP", Ylabel = "Tobacco"):
  fig2, ax2 = plt.subplots()
  colormap = np.array(['red', 'blue'])
  ax2.grid(visible = True, which = 'both')
  ax2.set_title(Subtitle)
  fig2.suptitle(Title)
  ax2.set_xlabel(Xlabel)
  ax2.set_ylabel(Ylabel)
  red_patch = mpatches.Patch(color='red', label='CHD Neg')
  blue_patch = mpatches.Patch(color='blue', label='CHD Pos')
  ax2.legend(handles=[red_patch,blue_patch], loc = "upper right")
  ax2.scatter(data[:range,0],data[:range,1], c = colormap[label[:range]]) 
  return
  
