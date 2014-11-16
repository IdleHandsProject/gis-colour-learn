"""
Kmeans clustering algorithm for colour detection in images

Initialise a kmeans object and then use the run() method.
Several debugging methods are available which can help to
show you the results of the algorithm.
"""

from PIL import Image
import random
import numpy
import numpy as np
import urllib, cStringIO
import os
import sys
import time
from urllib import FancyURLopener
import urllib2
import simplejson

from numpy import indices
import matplotlib.pyplot as plt
import mlpy
from PIL import Image, ImageFilter, ImageEnhance
from scipy.misc import imresize
from sklearn.svm import SVC
from sklearn import svm
from random import randint
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.learning_curve import learning_curve
from sklearn import preprocessing
from cv2 import *

from tempfile import TemporaryFile
outfile = TemporaryFile()


class Cluster(object):

    def __init__(self):
        self.pixels = []
        self.centroid = None

    def addPoint(self, pixel):
        self.pixels.append(pixel)

    def setNewCentroid(self):

        R = [colour[0] for colour in self.pixels]
        G = [colour[1] for colour in self.pixels]
        B = [colour[2] for colour in self.pixels]

        R = sum(R) / len(R)
        G = sum(G) / len(G)
        B = sum(B) / len(B)

        self.centroid = [R, G, B]
        self.pixels = []

        return self.centroid


class Kmeans(object):

    def __init__(self, k=1, max_iterations=5, min_distance=5.0, size=200):
        self.k = k
        self.max_iterations = max_iterations
        self.min_distance = min_distance
        self.size = (size, size)

    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)
        self.pixels = numpy.array(image.getdata(), dtype=numpy.uint8)

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        randomPixels = random.sample(self.pixels, self.k)

        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]

        iterations = 0

        while self.shouldExit(iterations) is False:

            self.oldClusters = [cluster.centroid for cluster in self.clusters]

            print iterations

            for pixel in self.pixels:
                self.assignClusters(pixel)

            for cluster in self.clusters:
                cluster.setNewCentroid()

            iterations += 1

        
        return np.array(cluster.centroid)
    def assignClusters(self, pixel):
        shortest = float('Inf')
        for cluster in self.clusters:
            distance = self.calcDistance(cluster.centroid, pixel)
            if distance < shortest:
                shortest = distance
                nearest = cluster

        nearest.addPoint(pixel)

    def calcDistance(self, a, b):

        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    def shouldExit(self, iterations):

        if self.oldClusters is None:
            return False

        for idx in range(self.k):
            dist = self.calcDistance(
                numpy.array(self.clusters[idx].centroid),
                numpy.array(self.oldClusters[idx])
            )
            if dist < self.min_distance:
                return True

        if iterations <= self.max_iterations:
            return False

        return True

    # ############################################
    # The remaining methods are used for debugging
    def showImage(self):
        self.image.show()

    def showCentroidColours(self):

        for cluster in self.clusters:
            image = Image.new("RGB", (200, 200), cluster.centroid)
            image.show()

    def showClustering(self):

        localPixels = [None] * len(self.image.getdata())

        for idx, pixel in enumerate(self.pixels):
                shortest = float('Inf')
                for cluster in self.clusters:
                    distance = self.calcDistance(cluster.centroid, pixel)
                    if distance < shortest:
                        shortest = distance
                        nearest = cluster

                localPixels[idx] = nearest.centroid

        w, h = self.image.size
        localPixels = numpy.asarray(localPixels)\
            .astype('uint8')\
            .reshape((h, w, 3))

        colourMap = Image.fromarray(localPixels)
        colourMap.show()
        


def main():
	if not os.path.exists('TrainingData.npz'):
		listsize = 10
		# Define search term
		
		classes = []
		training = np.zeros((listsize*5,3), dtype='uint8')
		searchTerm = ['purple','blue','red','green','yellow']
		
		for color in range(0,5):
			thisClass = np.empty(listsize)
			thisClass.fill(color)
			searchTerm = ['purple','blue','red','green','yellow']
			
		
			# Replace spaces ' ' in search term for '%20' in order to comply with request
			searchTerm = searchTerm[color].replace(' ','%20')


			# Start FancyURLopener with defined version 
			class MyOpener(FancyURLopener): 
				version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
			myopener = MyOpener()

			# Set count to 0
			count= 0
			

			##for i in range(0,5):
			i = 0
			while (count < listsize):    
				# Notice that the start changes for each iteration in order to request a new set of images for each loop
				url = ('https://ajax.googleapis.com/ajax/services/search/images?' + 'v=1.0&q='+searchTerm+'&start='+str(i*4)+'&userip=MyIP')
				print url
				request = urllib2.Request(url, None, {'Referer': 'testing'})
				response = urllib2.urlopen(request)

				# Get results using JSON
				results = simplejson.load(response)
				data = results['responseData']
				dataInfo = data['results']
			
				# Iterate for each result and get unescaped url
				for myUrl in dataInfo:
					if (count < listsize):
						count = count + 1
					print myUrl['unescapedUrl']
					X = myUrl['unescapedUrl']
					OnlineFile = cStringIO.StringIO(urllib.urlopen(myUrl['unescapedUrl']).read())
					##OnlineFile = cStringIO.StringIO(urllib.urlopen(X).read())
					try:
						image = Image.open(OnlineFile).convert('RGB')
						k = Kmeans()
						kmeanimg = k.run(image)
						myopener.retrieve(myUrl['unescapedUrl'],str(count)+'.jpg')
					except IOError:
						count = count - 1
					
					print kmeanimg
					training[count-1 + (color * listsize)] = kmeanimg

				# Sleep for one second to prevent IP blocking from Google
				print training
				time.sleep(0.5)
				i += 1
				 
			classes = np.append(classes, thisClass)
			classes = np.array(classes, dtype='uint8')
			##print totalList
			print classes


		X = training
		Y = classes
		np.savez('TrainingData', X=X, Y=Y)
		outfile.seek(0)
    
	npzfile = np.load('TrainingData.npz')
		
	X = npzfile['X']
	Y = npzfile['Y']
	#min_max_scaler = preprocessing.MinMaxScaler()
	#X = min_max_scaler.fit_transform(X)
	print X
	print Y
	time.sleep(2)
	print ("Creating Linear Graph")
	clf = svm.SVC(C=100, gamma=1, verbose=2, cache_size=1000)
	clf.fit(X, Y)
	#image = Image.open('10.jpg').convert('RGB')
	#print image
	while(1):
		raw_input("Press Enter to continue...")
		cam = VideoCapture(0)  #set the port of the camera as before
		retval, image = cam.read() #return a True bolean and and the image if all go right
		
		cam.release()
		print image
		#if retval:    # frame captured without any errors
		#	namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
		#	imshow("cam-test",image)
		#	waitKey(0)
		#	destroyWindow("cam-test")
		image = Image.fromarray(np.uint8(image)).convert('RGB')
		enhancer = ImageEnhance.Brightness(image)
		image = enhancer.enhance(20)
		k = Kmeans()
		result = k.run(image)
		print result
		predict = clf.predict([result])
		colour = ['purple','blue','red','green','yellow']
		print predict
		print colour[predict] 
	
	

	#    k.showImage()
	#    k.showCentroidColours()
	#    k.showClustering()
if __name__ == "__main__":
    main()
