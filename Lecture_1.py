#!/usr/bin/env python
# coding: utf-8

# In[130]:


# CMIT 2025 Python Lecture 1 == CMIT_P1.ipynb  [jupyter nbconvert --to python notebook.ipynb]
# 13 June 2025 9am      (If for PY, jupyter nbconvert --to script Lecture_1.ipynb)
''' Content in L1      [To launch juypter, type "python -m jupyter notebook"]
        1) Basic commands for scalar numbers and texts
        2) Programming basics:  
                (i) "if" coditional, (ii) "for" loop,  (iii) arrays, matrices, tensors
        3) HW Exercise 1
        '''
print('Hello CMIT participants :-)')
A = 2025.613


# Task 1 --- Basic commands for scalar numbers / print

# In[131]:


print('Task 1 --- Basic commands for scalar numbers / print')
a = 2025.5   # 1) Basic commands for scalar numbers and texts ----------
b = 6        # June
c = a / 5
d = 2025 % 10   #Compute remainder (0 divisible or not)
e = 11.34 // 5 # Integer Part
print('a=',a, ', b =', b,  ', c=a/5=', c,  
      ', d=2025 % 10 =',d, ',  e =  11.34 // 5 =',e,  '\n')


# Task 2 --- Basic commands for processing/extracting texts

# In[132]:


print('Task 2 --- Basic commands for processing/extracting texts')
F = 'Lecture_1.ipynb' # This refers to the file that we are using   [STRING]
a = F[0:3];  b=F[1:4];  c=F[-1:]    # Noting for ranges  0 start and  end-1   
What_is = F[-2]
print('\tStrings in F =', F)
print('a = F[0:3] =',a, ',  b=F[1:4]=',F[1:4], '  F[-2]=', 
      What_is, ' F[-2:] =', F[-2:], 'c=F[-1:]=',c)
a = F.split('.')[0]
b = F.split('.')[1]
print('String SPLIT 1 for F =',F, ': to', a, ' and ', b, ' or', F.split('.')[-1] )
C = F + '.zip'
print('Consider C =', C)
from os.path import splitext as Split
name, ext = Split( C )
print('String SPLIT 2 for C =',C, ': to', name, 'and ', ext)
print('String SPLIT 1 for C =',C, ': to', C.split('.')[0], 'and ', C.split('.')[1],' and',
       C.split('.')[2])


# Task 3 --- calling a built-in math function or defining own

# In[133]:


print('\nTask 3 --- calling a built-in math function or defining own')
import math
a = math.exp(1.0);  b=math.pi
print(a, b, math.sinh(1.0))

def my_sinh(x):
    y = math.exp(x);  ym = math.exp(-x)
    return (y-ym)/2.0
print(a, b, my_sinh(1.0), 'NB mine')


# Task 4 --- Programming basics:           (i) "if" coditional

# In[134]:


print('\nTask 4 --- Programming basics:           (i) "if" coditional')
import numpy as np    # NB for "if"  (a) :   (b) tab 111111111111111111111111111111
from numpy.random import rand as rd
x = rd() - 0.5
y = np.random.rand() - 0.5
print('x=',x, ' y=',y)
if x >= 0.0:
    print('non-negative x')
else:
    print('\tnegative x')

if y >= 0.1:
    print('positive y and y>=0.1')
else:
    print('\tless than 0.1')
print('|')

x1 = 'non-negative x' if  x >= 0.0 else '\tnegative x'  # Python Special 1-line
y1 = 'positive y and y>=0.1' if y >= 0.1 else '\tless than 0.1'
print(x1,' | ', y1)


# Task 5 --- Formatted print and Image loading

# In[ ]:


print('\nTask 5 --- Formatted print and Image loading')
a = r"C:\cmit2025\week1\File2.py";  b = 789;  c = math.pi
print(a, '\t| b=', b,'\t | c=', c)
print(a.split('\\')[-1], '\t\t\t| b=%04d' % b, '\t | c= %.4f' % c) # format old
print(f'{a.split('\\')[-1]} \t\t\t| b={b:04d} \t | c= {c:.4f}')    # format new
import os, cv2, matplotlib.pyplot as plt
my_image = 'Liver.jpg'
if os.path.exists( my_image ):
    A  = cv2.imread(my_image)    # default colour (n,n,3)
    A0 = cv2.imread(my_image, 0) # no colour so only (n,n)
fig = plt.figure(figsize=(8, 4)) 
ax1 = fig.add_subplot(1, 2, 1)     # Left subplot (similar to MATLAB)
ax1.imshow(A); ax1.set_title('Colour Image'); ax1.axis('off') # Hide axes if like
ax2 = fig.add_subplot(1, 2, 2)     # Right subplot
ax2.imshow(A0, cmap='gray') # cmap is optional as already (n,n)
ax2.set_title('Gray Image'); ax2.axis('off')
print('Images plotted - close it to proceed if in py');  plt.show()


# Task 6 --- Programming basics:           (ii) "for" loop

# In[136]:


print('\nTask 6 --- Programming basics:           (ii) "for" loop')

print('\tfor -- example 1 Taylor for exp(x)' )
n = 8;   s1 = 1.0;  x=0.123    ##   "for" loop
for i in range(1, n+1):
    s1 = s1 + x**i / math.factorial(i)   #or    s1 += x**i / math.factorial(i)  
print('s1 =', s1, ' where n =',n, ' True exp(x)=',math.exp(x))


# In[137]:


print('\n\tfor -- example 2 (efficient Taylor)')
s2 = 1;  x=0.123;  t=x;  n=25   # Efficient code // "if" for less printing
for j in range(1,n+1):
    if j % 5 == 0 or j==1:
        print('j=',j, ' term t_j=',t)
    s2 += t
    t = x*t/(j+1)       # note when j=1,  t=x*t/2 = x^2 / 2!
print('When x=', x, ',  s2 = %.8f' % s2, 'NB exp(x) = %.8f' % math.exp(x), ' er={:.2e}'.format(s2-math.exp(x)) )


# Task 7 --- Programming basics:           (iii) arrays, matrices, tensors

# In[138]:


print('\nTask 7 --- Programming basics:           (iii) arrays, matrices, tensors')
print('\n\t vectors a b c')
import numpy as np    # most common but there are other ways
a = np.arange(4)
b = np.arange(2,7)    # Moast identical to Matlab's linspace
print('a=',a, 'b=',b)
h = 0.125
c = np.arange(3.0, 4.0+h, h);  print(c)
print('Length: a b c =', len(a), len(b), len(c), ' and shape a =', a.shape)
ar = np.random.rand( 3,5 );  print('ar =\n', ar)
print('HERE ar shape =',ar.shape,' after conversion a`s shape =',
      a[:,np.newaxis].shape, 'or to row =',   a[np.newaxis,:].shape)
a1 = ar[0:2, 0:2];  a2=ar[-2:,-2:]
print('ar`s submatrices =\n', a1, ' and\n', a2)
at = np.random.rand( 3,5,2 ); 
print('HERE at (tensor) shape =',at.shape, ' after conversion at`s shape =',
       at[np.newaxis,:].shape)


# In[139]:


print('pytorch has its own set of commands similar to nunmpy')
import torch
t1= torch.rand( () )
ta = torch.rand( (3,5) )
print(t1,'\n', ta)
print('Minor point: np.random.seed(42) and torch.manual_seed(42) cannot agree on rd')
print('Usually we use: torch.from_numpy or  ta.numpy()')
import datetime
time0 = datetime.datetime.now().strftime("%Y %m %d @ %H:%M:%S")
print('\t time =', time0)
# HOME WORK 1 BELOW -- If not sure about the maths concepts, check wiki or other sources


# In[140]:


s=r'\nTASK:  If not sure about the maths concepts, check wiki or other sources.\n\
   Write a code to generate the Hilbert matrix $H$ with $H_{ij}=1/(ij-1)$ for $i,j=1,...,n$.\n\
   Then compute its condition number $cond(H)$ for $n=4, 8$ by numpy.linalg.cond\n\
   If your code runs okay,  just show your answers on screen in the next lecture'
print('HW 1 - CMIT 2025\n',s) # if run in py


# Home work 1 --- Generate a matrix $H$ and compute its condition number

# 
# TASK:  If not sure about the maths concepts, check wiki or other sources.\n\
#    Write a code to generate the Hilbert matrix $H$ with $H_{ij}=1/(ij-1)$ for $i,j=1,...,n$.
# 
#    Then compute its condition number $cond(H)$ for $n=4, 8$ by numpy.linalg.cond
#    If your code runs okay,  just show your answers on screen in the next lecture
# 
