REQUIREMENT:
	- Python 3.
	- Python packages: pandas, numpy, scipy, chardet, collections, operator.

HOW TO RUN:

	python netflix.py 

 + ratings.txt: file stores rating of all users to items in the database. (Format: ItemID, userID, rate)
 + movie_titles.txt: file contains movie names. (Format: movieID, year-of-release, movie name)

FEATURES:
	- The program can give predicted score of user U to item I.
		+ Input: userID, movieID
		+ Output: Predicted rating.

	- The program can give recommendation movie M to user U.
		+ Input: userID, year-of-release.
		+ Output: recommend a movie for that user for that year-of-release.


WALKTHROUGH:
	1) Predict Score:
		- User can input K for number of nearest neighbors using in predicting scores. Default value: K = 4.
		- Predicted Scores will be recomputed every time a new value of K is entered.
		- The program selects K neighbors with highest similarity weights from the matrix.
		- From these K neighbors, only ratings from those neighbors who rated item I will be used to compute predicted score of user U to item I.
		- Predicted Scores will be saved in txt file.

	2) Recommendation system:
		- The program requires userID and year-of-movie.
		- The program will recommend a movie M in the input year to user U, which has the highest
		  predicted score among all movies in that year that the user has not yet reviewed.
		* Note: if movieID in ratings.txt and movie_titles.txt does not match, the program can not find the movie; therefore, it assumes that the user has watched and rated all movies in database.

	


