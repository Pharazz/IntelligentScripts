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
  # ax1.set_xlim([0,10])
  for i in range(len(ClusterList)):
    ax1.scatter(ClusterList[i].getPoints()[:,0],ClusterList[i].getPoints()[:,1], c = colorsList[i], label=('Cluster' + str(i)))
    ax1.scatter(ClusterList[i].center[0], ClusterList[i].center[1], marker = "x", c = 'black')
  ax1.legend()
  return
  
def plotClustersKNN(ClusterList, Title = "K-NN Clustering", Subtitle = "Epoch = 1", Xlabel = "X", Ylabel = "Y"):
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

def getStats2(ClusterList, points, labels):
  GuessList = getGuesses(ClusterList, points)
  for i in range(len(GuessList)):
    if GuessList[i] > 1:
      GuessList[i] = 1
  [TruePos, TrueNeg, FalsePos, FalseNeg] = [0,0,0,0]
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

def getDistMat(npl):
  tmp = np.zeros((len(npl),len(npl)))
  for i in range(len(npl)):
    for j in range(i+1, len(npl)):
      dist = np.linalg.norm(npl[i]-npl[j])
      tmp[i,j] = dist
      tmp[j,i] = dist
  return tmp

def stdData(data):
  data_norm = (data-data.mean())/data.std()
  print("STD:" + str(data.std()))
  print("MEAN:" + str(data.mean()))
  return data_norm


