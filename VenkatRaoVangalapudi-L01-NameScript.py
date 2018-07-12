#  script to return my  name
#  and print time

def ReturnMyName():
    """Retuens "Venkat Rao Vangalapudi" """
    return "Venkat Rao Vangalapudi"

# call ReturnMyName 
myname = ReturnMyName()

#print myname
print (myname)

# [1]: https://www.w3resource.com/python-exercises/python-basic-exercise-3.php
"""import datetime package """
import datetime

"""asign current date and time to now variable """
now = datetime.datetime.now()

print ("Current date and time : ")

"""Print Date Time """
print (now.strftime("%Y-%m-%d %H:%M:%S"))