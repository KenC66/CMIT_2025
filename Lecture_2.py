#!/usr/bin/env python
# coding: utf-8

# Ken's Lecture 2 (Python) -- CMIT summer internship programme

# In[1]:


#                       (To convert to PY, jupyter nbconvert --to script Lecture_2.ipynb)
print('Following Task 6 from Lecture 1 i.e. for -- example 2 (efficient Taylor)')
import math
s2 = 1;  x=0.123;  t=x;  n=25   # Efficient code // "if" for less printing
for j in range(1,n+1):
    if j % 15 == 0 or j==1:
        print('j=',j, ' term t_j=',t)
    s2 += t
    t = x*t/(j+1)       # note when j=1,  t=x*t/2 = x^2 / 2!
print(f'Answer s2 = {s2: .6f} vs True {math.exp(x): .6f}')


# Recall the math task was to compute the sum $s = \displaystyle \sum_{j=0}^n \frac{x^j}{j!}$.

# Task 8  from "def" to advanced "class" - a touch of learning set up

# In[2]:


print('Task 8  from "def" to advanced "class" - a touch of learning set up')
print("Here we start with the basic form: {def}, as with MATLAB`s {function}")
def my_sum(x,n):
    s2 = 1;   t=x;   # Efficient code // "if" for less printing
    for j in range(1,n+1):
        s2 += t
        t = x*t/(j+1)       # note when j=1,  t=x*t/2 = x^2 / 2!
    return s2

x=0.123; answer = my_sum(x, 25 )
print(f'Answer s2 = {answer: .6f} vs True {math.exp(x): .6f}')


# In[3]:


print('advanced "class" form 1 - a touch of learning set up')
class my_sum1():
    def __init__(talk, x=1.0,n=10):
        talk.x=x
        talk.n=n
    def comp(talk):
        s1 = 1;   t=talk.x;   # Efficient code // "if" for less printing
        for j in range(1,talk.n+1):
            s1 += t
            t = x*t/(j+1)       # note when j=1,  t=x*t/2 = x^2 / 2!
        return s1
answer1 = my_sum1(x=0.789, n=15 ).comp() # New -- never seen before
x=0.123; answer = my_sum1(x, 25 ).comp()
print(f' Answer = {answer: .6f} vs True {math.exp(x): .6f}')


# In[4]:


print('advanced "class" form 2 - a touch of learning set up')
class my_sum2():
    def __init__(talk, x=1.0):
        talk.x=x
    def comp(talk,n=10):
        s2 = 1;   t=talk.x;   # Efficient code // "if" for less printing
        for j in range(1,n+1):
            s2 += t
            t = x*t/(j+1)       # note when j=1,  t=x*t/2 = x^2 / 2!
        return s2
answer2 = my_sum2(x=0.789).comp(n=10) # New -- never seen before
x=0.123; answer = my_sum2(x ).comp(25)
print(f' Answer s2 = {answer: .6f} vs True {math.exp(x): .6f}')


# In[4]:


print('advanced "class" form 3 - a touch of learning set up')
class my_sum3():
    def __init__(talk):  # Here Outer has no paras so "pass" to make syntax happy
        pass
    def comp(talk,x=1.2, n=10):
        s3 = 1;   t=x;   # Efficient code // "if" for less printing
        for j in range(1,n+1):
            s3 += t
            t = x*t/(j+1)       # note when j=1,  t=x*t/2 = x^2 / 2!
        return s3
answer3 = my_sum3().comp(n=10,x=0.789) # NB named can swap but unnamed not
x=0.123; answer = my_sum3().comp(x,25)
print(f' Answer s3 = {answer: .6f} vs True {math.exp(x): .6f}')


# Task 9 -- Reading all images from a directory via "glob"

# In[ ]:


import warnings
warnings.simplefilter("ignore", UserWarning)

import glob, os, sys
from skimage import io
import matplotlib.pyplot as plt
print('Below we first set up a`wildcard` string. Then let glob find all names.')
str_img1 = 'Cells/train/image/*.png' 
str_lab1 = 'Cells/train/label/*.png' 
img_dir = os.path.dirname(str_img1)
if not os.path.exists(img_dir):
    print('Directory:', img_dir, ' does not exist yet')
files_1  = sorted(glob.glob( str_img1, recursive=True) ) # Images (not read yet)
Count = len(files_1)
print('Found %d img files' % Count)

lab_dir = os.path.dirname(str_lab1)
try:
    os.makedirs(lab_dir)
    print('Directory:', lab_dir, ' does not exist yet')
except:
    pass


files_2  = sorted(glob.glob( str_lab1, recursive=True) ) # Labels
Coun2 = len(files_2)
if not (Count == Coun2):
    print('Files may not match -- check')
print('\tLabel files =', Coun2 )
for i in range(Count):
    imgs = io.imread(files_1[i], 0)
    if (i+1) % 14 ==0:
        plt.imshow( imgs );  plt.title('Image %d' % i)
        plt.draw(); plt.pause(1) 

plt.figure()
for i in range(0,Count,7):
    imgs = io.imread(files_1[i], 1)
    labs = io.imread(files_2[i], 0)
    plt.subplot(1,2,1);  plt.imshow( imgs );  plt.title('Image %d' % i)
    plt.subplot(1,2,2);  plt.imshow( labs );  plt.title('Label %d' % i)
plt.show()

######## 
# %%
print('\nHome work 2 --- Generate a matrix and compute its condition number\n')

s='TASK:  Use the same Hilbert matrix as Home work 1.\n\
   Write a code using "class" to generate a matrix $A = H + sigma*I$ \n\
   where\n\
   $H_{ij}=1/(i+j-1)$ for $i,j=1,...,n$,  \n\
   sigama is a scalar and I=identity matrix.\n\
   Then compute its condition number $cond(A)$ for the cases of \n\
    (i) sigma=0.5 and n=4\n\
   (ii) sigma=0.5 and n=8\n\
   If your code runs okay,  just show it quickly and your answers on screen in the next lecture\n'
print(s)

import datetime
time0 = datetime.datetime.now().strftime("%Y %m %d @ %H:%M:%S"); print('\t time =', time0)
