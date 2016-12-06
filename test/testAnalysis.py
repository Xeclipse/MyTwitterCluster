labelsFile='../resource/fsd/labels'
predictLabelsFile='../resource/fsd/labels'
file=open(labelsFile)
labels=[int(i)-1 for i in file.readlines()]
file.close()

file=open(labelsFile)