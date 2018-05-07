__author__ = "Diem-Trang Pham"

import sys
import os
import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import chardet
from collections import OrderedDict
from operator import itemgetter

class CF(object):
	def __init__(self, ratings, df_movie_titles, rated_items, K=4):
		self.ratings = ratings
		self.K = K
		self.rated_items = rated_items
		self.df_movie_titles = df_movie_titles

		self.avg_rating = {}
		self.normRate = {}
		self.similarity = {}
		self.predict = {}
		self.not_rated = {}
		self.commonItems = self.getCommonItems()

	def setK(self, K):
		self.K = K

	def __computeAvgRating(self):
		avg_rating = {}
		for u in self.ratings:
			avg_rating[u] = sum(self.ratings[u].values())/len(self.ratings[u].values())
		
		self.avg_rating = avg_rating
		return

	def __normalizeRating(self):
		normRate = {}
		for u in self.ratings:
			normRate[u] = {}
			for i in self.ratings[u]:
				normRate[u][i] = self.ratings[u][i] - self.avg_rating[u]

		self.normRate = normRate
		return

	# compute similarity weight between user1 and user2 as correlation coefficient (Pearson's coefficient)
	# w(a,u) = (sigma ( r(a,i) -r_bar(a) ) * ( r(u,i) - r_bar(u) ) ) / sqrt( ( sigma ( r(a,i) - r_bar(a) )^2 ) * (  sigma ( r(u,i) - r_bar(u) )^2  )  )
	# I is the set of items rated by both user a and user u.
	# r(u,i) is the rating given by user u to item i.
	# r_bar(u) is the mean rating of u across items in I.
	def __sim_weight(self, user1, user2):
		w = None
		
		sum_nR = 0
		sum_sqr_nR1 = 0
		sum_sqr_nR2 = 0

		for item in self.commonItems[user1][user2]:
			nR1 = self.normRate[user1][item]
			nR2 = self.normRate[user2][item]
			sqr_nR1 = nR1*nR1
			sqr_nR2 = nR2*nR2

			sum_nR = sum_nR + nR1*nR2
			sum_sqr_nR1 = sum_sqr_nR1 + sqr_nR1
			sum_sqr_nR2 = sum_sqr_nR2 + sqr_nR2

			try:
				w = sum_nR / math.sqrt(sum_sqr_nR1*sum_sqr_nR2)
			except ZeroDivisionError:
				w = float('NaN')
					
		if w == None: # no common item
			return float('NaN')
		else:
			return w

	# count the number of common items from every pair of users and keep in a dictionary
	def getCommonItems(self):
		commonItems = {}
		for user1 in self.ratings:
			commonItems[user1] = {}
			for user2 in self.ratings:
				if user1 != user2:

					intersection = self.ratings[user1].keys() & self.ratings[user2].keys()
					commonItems[user1][user2] = intersection
					if user2 not in commonItems:
						commonItems[user2] = {}
					commonItems[user2][user1] = intersection

		return commonItems

	def similarity_matrix(self):
		print('\n Computing similarity weight matrix....')
		self.__computeAvgRating()
		self.__normalizeRating()

		sim = {}
		for user1 in self.ratings:
			if user1 not in sim:
				sim[user1] = {}	

			for user2 in self.ratings:
				if user2 not in sim:
					sim[user2] = {}

				if user1 != user2:
					if user1 in sim[user2]:
						sim[user1][user2] = sim[user2][user1]
					
					if user2 in sim[user1]:
						sim[user2][user1] = sim[user1][user2]

					if user2 not in sim[user1] and user1 not in sim[user2]:
						w = self.__sim_weight(user1, user2)
						sim[user1][user2] = w
						sim[user2][user1] = w

		sim = self.__sort_DictByValues(sim)
		self.similarity = sim

		self.saveWMatrix2File()
		return

	# sort dict of dict by values
	def __sort_DictByValues(self, d):
		for user in d:
			d[user] = OrderedDict(sorted(d[user].items(), key = itemgetter(1), reverse = True))

		return d

	# select k nearest neighbors who rated item i for user u
	def __get_Neighbors_rated_i(self, u, i):

		neighbors = self.similarity[u]

		neighbors_rated_i = []
		for neighbor in neighbors:
			if i in self.ratings[neighbor]:
				neighbors_rated_i.append(neighbor)

		return neighbors_rated_i[:self.K]

	# compute predicted rate
	# Prediction of user a's rating to item i
	# p(a,i) = r_bar(a) + sigma_uinK ( r(u,i) -r_bar(u) )*w(a,u)/ sigma|w(a,u)|
	# r_bar(u) is the average rating for u
	# w(a,u) is the similarity weight between user a and user u
	# K is the set of nearest neighbors for user a based on w(a,u)
	def predictRating(self):
		print('\n Predicting scores....')

		predict = {}
		not_rated = {}
		for user in self.ratings:
			predict[user] = {}
			not_rated[user] = {}
			for item in self.rated_items:	
				neighbors = self.__get_Neighbors_rated_i(user, item)	

				sum_w = 0.0
				sum_rw = 0.0
				for neighbor in neighbors:
					if item in self.normRate[neighbor]:
						w = self.similarity[user][neighbor]
						if np.isnan(w):
							w = 0
						abs_w = np.abs(w)
						
						sum_rw = sum_rw + self.normRate[neighbor][item]*w
						sum_w = sum_w + abs_w

				predict[user][item] = self.avg_rating[user] + sum_rw/sum_w

				# keep track of unrated items for user 
				if item not in self.ratings[user]:
					if item not in not_rated[user]:	
						not_rated[user][item] = predict[user][item]

		self.predict = predict
		self.not_rated = self.__sort_DictByValues(not_rated)

		# print('Not Rated.')
		# print_dictOfDict(self.not_rated)

		self.savePredictedScore2File()
		return 

	# save Similarity Matrix to txt file
	def saveWMatrix2File(self):
		SCRIPT_DIR = os.getcwd()
		wmatrix_file = os.path.join(SCRIPT_DIR,'wmatrix_K'+str(self.K)+'.txt')
		with open(wmatrix_file, 'w') as f:
			f.write(str(self.similarity))

		print("Similarity Matrix is saved at :", wmatrix_file)

		return

	# save Predicted Score to txt file
	def savePredictedScore2File(self):
		SCRIPT_DIR = os.getcwd()
		predict_file = os.path.join(SCRIPT_DIR,'predictions_K'+str(self.K)+'.txt')
		with open(predict_file, 'w') as f:
			for user in self.predict:
				for item in self.predict[user]:
					if np.isnan(self.predict[user][item]):
						f.write("User = "+str(user)+" : Movie = "+str(item)+"\t"+ str(self.avg_rating[user])+"\n")	
					else:
						f.write("User = "+str(user)+" : Movie = "+str(item)+"\t"+ str(self.predict[user][item])+"\n")
			
		print("Prediction Score is saved at :", predict_file)
		return 

	# recommend a movie of input year for user
	def recommend(self, user, year):
		movie_titles = df_movie_titles.set_index('movieID').T.to_dict('list')
		movie_user_year = self.df_movie_titles[self.df_movie_titles['YearOfRelease'] == year]

		mids = list(movie_user_year['movieID'])
		if len(mids) == 0:
			print('No movie of year', year, 'in database.')
			return

		# this part can not be used if movieID in movie_titles.txt does not match with movieID in ratings.txt
		for item in self.not_rated[user]:
			if item in mids:
				name = movie_titles[item][1]
				print('\n ---> Recommend movie [', name, '] for user', user)
				return

		print('User ', user, 'has watched and rated all movies in this year.')
		return

	def fit(self):
		self.similarity_matrix()
		self.predictRating()
	
	def printNotRate(self, user):
		print('\nItem not rated by user', user,':')

		if len(self.not_rated[user]) == 0:
			print('User', user, 'has rated all items.')
			return

		for item in self.not_rated[user]:
			print('Item:',item, '\tScore:', self.not_rated[user][item])

		return

	def userPredictedScore(self, user, item):
		if user in self.predict:
			if item in self.predict[user]:
				if np.isnan(self.predict[user][item]):
					print('\n ---> User:',user,'\tMovie:',item,'\tScore:',self.avg_rating[user])	 
				else:
					print('\n ---> User:',user,'\tMovie:',item,'\tScore:',self.predict[user][item])
		return

def readRatings(file):
	ratings = {}
	rated_items = []
	with open(file, 'r') as f:
		for line in f:
			line = line.strip()
			mID, uID, rate = line.split(',')

			if uID not in ratings:
				ratings[uID] = {}
			ratings[uID][mID] = float(rate)

			if mID not in rated_items:
				rated_items.append(mID)
	return ratings, rated_items

def print_dictOfDict(d):
	for u in d:
		for i in d[u]:
			print(u, i, d[u][i])

def printDict(d):
	for u in d:
		print(u, d[u])

def menu():
	print("\n\n############################")
	print("1. Enter number of neighbors (K).")
	print("2. Print Predicted Score of a user to an item.")
	print("3. Recommendation system.")
	print("4. Print Item(s) not rated by a user.")
	print("5. Exit.")
	print("############################\n\n")

if __name__ == '__main__':
	# if len(sys.argv) != 3:
	# 	print("USAGE: ", sys.argv[0], " ratings.txt movie_titles.txt")
	# 	sys.exit(0)

	# rating_file = sys.argv[1]
	# movie_file = sys.argv[2]
	rating_file = './data/test.txt'
	movie_file = './data/movie_titles.txt'

	print('\nReading database...')
	# read ratings
	ratings, rated_items = readRatings(rating_file)

	# read movie_titles
	with open(movie_file, 'rb') as f:
	    result = chardet.detect(f.read())
	df_movie_titles = pd.read_csv(movie_file, header=None, names=['movieID', 'YearOfRelease', 'Title'],encoding=result['encoding'], converters={'movieID': str})

	# # build model
	print('\nInitializing Collaborative Filtering ...')
	rs = CF(ratings, df_movie_titles, rated_items)
	rs.setK(4)
	rs.fit()

	print('-----------------------------------')
	
	while True:
		menu()
		choice = input("Please select : ")
		if choice == '1':
			K = int(input("Enter K: "))
			if K <= 0:
				print("Invalid K.")
				exit()
			rs.setK(K)
		elif choice == '2':
			userID = input("Enter userID: ")
			movieID = input("Enter movieID: ")
			rs.predictRating()
			rs.userPredictedScore(userID, movieID)

		elif choice == '3':
			print('Recommendation system.')
			print(' If you did not enter K, the default value of K is 5.')
			userID = input("\nEnter userID: ")
			year = int(input("Enter Year for movie: "))
			rs.recommend(userID, year)
		elif choice == '4':
			userID = input("Enter userID: ")
			rs.printNotRate(userID)

		elif choice == '5':
			print("Exit!")
			exit()
		else:
			print("Wrong option.")


