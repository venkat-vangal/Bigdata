# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:29:51 2018

@author: v-venva
"""

# Use the pandas package
import pandas as pd
# The url for the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
Adult = pd.read_csv(url, header=None)
Adult.columns = ['MostRecent', 'Donations', 'Volume', 'First', 'Donated']
Adult.head()

import pandas as pd
# The url for the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"
Which of the following assignments is appropriate for reading all the data into a data frame?

	
BloodDonation = pd.read_csv(url, header=0)
 	none of these options
	
BloodDonation = pd.read_csv(url, header=None)
 	
BloodDonation = pd.read_csv(url, header=False)
 	
BloodDonation = pd.read_csv(url)
 
Flag this Question
Question 4 1 pts
Consider the following code:

# Use the pandas package
import pandas as pd
# The url for the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#Which of the following assignments is appropriate for reading all the data into a data frame?


	
Adult = pd.read_csv(url, header=None)
Adult.head()
 	
Adult = pd.read_csv(url, header=False)

# Use the pandas package
import pandas as pd
# The url for the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
Adult = pd.read_csv(url)
Adult.head()
 	
Adult = pd.read_csv(url, header=None)
 	
Adult = pd.read_csv(url, header=0)
No new data to save. Last checked at 7:35pm 
Questions
AnsweredQuestion 1
AnsweredQuestion 2
Haven't Answered YetQuestion 3
Haven't Answered YetQuestion 4
Time Elapsed: Hide
Attempt due: Apr 16 at 7pm

