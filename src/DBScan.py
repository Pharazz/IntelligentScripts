class DBSCluster:
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
  def __str__(self):
    return "---\n DBS-Cluster: " + self.name + "\nContains: : {} points\n".format(len(self.points)) + str(self.points) +"\n---" 

def DBScan(points, epsilon = 2, Min_Samples = 2):
  SetList = []
  Clusters = [None]*len(points)
  SetPntIDX = set(range(len(points)))
  #Get Distance Matrix
  distMat = getDistMat(points)
  #For each point Trim Distance Matrix to be in epsilon
  for i in range(len(points)):
    loc = np.squeeze(np.where(distMat[i][:] <= epsilon)).tolist()
    if type(loc) != list: #Only sets that meet minimum requirements
      continue
    elif len(loc) >= Min_Samples:
      SetList.append(set(loc))
  #Now have 1D list of N sets of point indexes within epsilon
  ClusterCount = 0
  SetClusteredPoints = set()
  while len(SetList) > 0:
    try: 
      SetA = SetList.pop()
      SetList = list(filter(lambda x: x != SetA, SetList)).copy()
    except:
      print("No more Sets to Pop\nSet List Length = {}".format(len(SetList)))
      break
    for i in range(len(SetList)):
      for SetN in SetList:
        if SetA.isdisjoint(SetN) == False:
          print("Set A {} intersects Set N {}".format(SetA, SetN))
          SetA.update(SetN)
          SetList = list(filter(lambda x: x != SetN, SetList)).copy()
        else:
          print("Set A {} does not intersect Set N {}".format(SetA, SetN))
          continue
    Clusters[ClusterCount] = copy.deepcopy(DBSCluster("Cluster"+str(ClusterCount), np.squeeze([points[list(SetA)]])))
    ClusterCount+=1
    SetClusteredPoints.update(SetA)
    print("Found Set:{}".format(SetA))
    print("{} remaining Sets in list: {}".format(len(SetList),SetList))
  #Now get Outliers
  Outliers = SetPntIDX.difference(SetClusteredPoints)
  print("Outliers: {}".format(Outliers))
  print("Total Count Outliers: {}".format(len(Outliers)))
  print("Total Count Clustered: {}".format(len(SetClusteredPoints)))
  if Outliers:
    Clusters[ClusterCount] = copy.deepcopy(DBSCluster("Outliers", np.squeeze([points[list(Outliers)]])))
  filtered_items = filter(lambda item: item is not None, Clusters)
  new_Clusters = list(filtered_items)
  return new_Clusters

def getStats2(ClusterList, points, labels):
  GuessList = getGuesses(ClusterList, points)
  for i in range(len(GuessList)):
    if GuessList[i] > 1:
      GuessList[i] = 1
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
