class KMeanCluster:
  def __init__(self, name = "Cluster1", center=[], points=[]):
    self.name = name
    self.center = center
    self.points = points
  def rename(self, name):
    self.name = name
  def setCenter(self, center):
    self.center = center
  def setPoints(self, points):
    self.points = points
  def getNewCenter(self):
    self.center = [sum(x)/len(x) for x in zip(*self.points)]
  def removeFromCluster(self, point):
    self.points.remove(point)
  def addToCluster(self, point):
    self.points.append(point)
  def emptyCluster(self):
    self.points.clear()
    self.center.clear()
  def emptyClusterPoints(self):
    self.points.clear()
  def getPoints(self):
    return np.array(self.points)
  def __str__(self):
    return "---\n K-Means Cluster: " + self.name +"\nCenter: @" + str(self.center) +"\nContains: " + str(self.points) +"\n---"

def KMeansAlg(Seeds, Points=[], epochs = 1):
  Clusters = [None] * len(Seeds)
  NumOfClusters = len(Seeds) #Get number of Clusters
  for i in range(NumOfClusters):
    #Make copy of clusters add them to list
    Clusters[i] = copy.deepcopy(KMeanCluster("Cluster"+str(i), Seeds[i]))
  #Run for however many epochs
  for k in range(epochs):
    [Clusters[i].emptyClusterPoints() for i in range(NumOfClusters)]
    for i in range(len(Points)):
      #Initialize distances to zero
      dist = np.zeros(NumOfClusters)
      for j in range(NumOfClusters):
        #Distance from Cluster Center to Point
        dist[j] = np.linalg.norm(Clusters[j].center-Points[i])
      #Get index of Minimum distance
      ChosenCluster = np.argmin(dist, axis = 0)
      #Append point to cluster
      Clusters[ChosenCluster].addToCluster((Points[i]))
      # Clusters[ChosenCluster].addToCluster(list(Points[i]))
    for i in range(NumOfClusters):
      #Get new center for cluster
      Clusters[i].getNewCenter()
  return Clusters

def PrintClassList(ClassList):
  #How to Print Items in List
  FullPrint = [print(item) for item in ClassList]
  print(FullPrint)

def plotClusters(ClusterList, Title = "K-Means Clustering", Subtitle = "Epoch = 1", Xlabel = "X", Ylabel = "Y"):
  colorArray = ['red', 'blue', 'green', 'orange', 'pink',]
  colorsList = ["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"]
  fig1, ax1 = plt.subplots()
  ax1.grid(visible = True, which = 'both')
  ax1.set_title(Subtitle)
  fig1.suptitle(Title)
  ax1.set_xlabel(Xlabel)
  ax1.set_ylabel(Ylabel)
  for i in range(len(ClusterList)):
    ax1.scatter(ClusterList[i].getPoints()[:,0],ClusterList[i].getPoints()[:,1], c = colorsList[i], label=('Cluster' + str(i)))
    ax1.scatter(ClusterList[i].center[0], ClusterList[i].center[1], marker = "x", c = 'black')
  ax1.legend()
  return
  
def getGuesses(ClusterList, points):
  GuessList = []
  for i in range(len(points)):
    for j in range(len(ClusterList)):
      if points[i] in ClusterList[j].getPoints():
        GuessList.append(j)
        break
  return np.array(GuessList)

def getStats(ClusterList, points, labels):
  GuessList = getGuesses(ClusterList, points)
  [TruePos, TrueNeg, FalsePos, FalseNeg] = [0,0,0,0]
  # [sum(x)/len(x) for x in self.points)]
  for i, (data, label) in enumerate(zip(GuessList, labels)):
    if ((data == label) and data == 1):
      TruePos+=1
    elif ((data == label) and data == 0):
      TrueNeg+=1
    elif ((data != label) and data == 1 and label == 0):
      FalsePos+=1
    else:
      FalseNeg+=1
  print("True Positive: {}".format(TruePos) + "\nTrue Negative: {}".format(TrueNeg) +
        "\nFalse Positive: {}".format(FalsePos)+ "\nFalse Negative: {}".format(FalseNeg))
  Acc = (TruePos+TrueNeg)/(len(labels))
  Prec = TruePos/(TruePos+FalsePos)
  Recall = TruePos/(TruePos + FalseNeg)
  F1Score = 2*((Prec*Recall)/(Prec+Recall))
  print("Accuracy: {}".format(Acc) + "\nPrecision: {}".format(Prec) +
        "\nRecall: {}".format(Recall)+"\nF1-Score: {}".format(F1Score))
  return 
