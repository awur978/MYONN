
# ==============================================
# ALL THE ACTIVATION FUNCTIONS* FROM WIKIPEDIA

#                 * with a corresponding graph

# Charlie Ann Fornaca | cup-of-char.com
# ==============================================

# Wikipedia article: wikipedia.org/wiki/Activation_function

# alpha (a) = "a stochastic variable sampled from a uniform distribution at 
#   training time and fixed to the expectation value of the distribution at 
#   test time". Some of these functions take a second argument for alpha. 
#   Default alpha is 0.01.
  
# sigma (s) = logistic function.

# Probability given x (p) = Currently set to 1 by default in Soft Clipping 
#   derivative.

# Imports:
import numpy as np
import scipy.special
from math import e,sqrt,sin,cos

# Identity function: 
def identity(x):
  return x

# Identity function derivative:
def identity_deriv(x):
  return 1

# Binary step function:
def step(x):
  return 0 if x < 0 else 1
  
# Sigmoid function using SciPy: 
def expit(x):
  return scipy.special.expit(x)

# Sigmoid/logistic functions with Numpy:
def logistic(x):
  return 1/(1 + np.exp(-x))

# Sigmoid/logistic function derivative:
def logistic_deriv(x):
  return logistic(x)*(1-logistic(x)) 

# Tanh/Hyperbolic Tangent Activation Function:
def tanh(x):
  return np.tanh(x)

# Tahn function derivative:
def tanh_deriv(x):
  return 1.0 - np.tanh(x)**2
  
# ArcTan function:
def arctan(x):
  return np.arctan(x)

# ArcTan function derivative:
def arctan_deriv(x):
  return 1/((x**2)+1)

# Softsign function:
def softsign(x):
  x = abs(x)
  return x/(1+x)

# Softsign function derivative:
def softsign_deriv(x):
  x = abs(x)
  return 1/((1+x))**2

# Inverse square root unit (ISRU):
def isru(x, a=0.01):
  return x/sqrt(x+(a*(x**2)))

# ISRU derivative:
def isru_deriv(x, a=0.01):
  return (x/sqrt(1+(a*(x**2))))**3

# Inverse square root LINEAR unit (ISRLU):
def isrlu(x,a=0.01):
   return x/sqrt(1+(a*(x**2))) if x < 0 else x

# ISRLU derivative:
def isrlu_deriv(x,a=0.01):
  return (1/sqrt(1+(a*(x**2))))**3 if x < 0 else 1

# Square Nonlinearity (SQNL):
def sqnl(x):
  if x < (-2.0): 
    return -1
  elif x < 0.0:
    return x+((x**2.0)/4.0)
  elif x <= 2.0:
    return x-((x**2)/4.0)
  elif x > 2.0:
    return 1
  
# SQNL derivative:
def sqnl_deriv(x):
  # Note: This function returns two values.
  return (1-(x/2), 1+(x/2))
 
# ReLu (Rectified Linear Unit) function:
def relu(x):
  return 0 if x < 0.0 else x
  
# ReLU derivative:
def relu_deriv(x):
  return 0 if x < 0.0 else 1
    
# Leaky ReLU:
def leaky(x):
  return x*0.01 if x < 0 else x
      
# Leaky ReLU derivative:
def leaky_deriv(x):
  return 0.01 if x < 0 else 1

# Parametric rectified linear unit (PReLU):
def prelu(x,a=0.01):
  return x*a if x < 0 else x  
  
# PReLU derivative:
def prelu_deriv(x,a=0.01): 
  return a if x < 0 else 1

# Randomized leaky rectified linear unit (RReLU):
def rrelu(x,a=0.01):
  return a*x if x < 0 else x
  
# RReLU derivative:
def rrelu_deriv(x,a=0.01):
  return a if x < 0 else 1
  
# Exponential linear unit (ELU):
def elu(x,a=0.01):
  return a*(((e)**2)-1) if x <= 0 else x
  
# ELU derivative:
def elu_deriv(x,a=0.01):
  return elu(x,a)+a if x <= 0 else 1
  
# SoftPlus
def softplus(x):
  return np.log(1+((e)**x))

# SoftPlus derivative: 
def softplus_deriv(x):
  return 1/(1+((e)**-x))

# Bent identity:
def bentid(x):
  return (sqrt(((x**2)+1)-1)/2)+x

# Bent identity derivative:
def bentid_deriv(x):
  return (x/(2*(sqrt((x**2)+1))))+1

# SoftExponential:
def softex(x,a=0.01):
  if a < 0:
    return -((np.log(1-a*(x+a)))/a)
  elif a == 0:
    return x
  elif a > 0:
    return (((e)**(a*x))/a)+a
  
# SoftExponential derivative:
def softex_deriv(x,a=0.01):
  return 1/(1-a*(a+x)) if a < 0 else (e)**(a*x)
  
# Soft Clipping:
def softclip(x,a=0.01):
  return (1/a)*(np.log10((1+((e)**(a*x)))/(1+(e)**(a*(x-1)))))
  
# Soft Clipping derivative.
def softclip_deriv(x,a=0.01,p=1):
  def sech(x):
    return np.cosh(x)**(-1)
  return (0.5)*(np.sinh(p/2))*(sech((p*x)/2))*sech((p/2)*(1-x))
   
# Sinusoid:
def sinusoid(x):
  return sin(x)
  
# Sinusoid derivative:
def sinusoid_deriv(x):
  return cos(x) 
  
# Sinc:
def sinc(x):
  return 1 if x == 0 else (sin(x))/x
  
# Sinc derivative:
def sinc_deriv(x):
  return 0 if x == 0 else ((cos(x))/x)-((sin(x))/x**2)              
  
# Gaussian: 
def gaussian(x):
  return (e)**((-x)**2)
   
# Gaussian derivative:
def gaussian_deriv(x):
    return -2*x*(e)**((-x)**2)
