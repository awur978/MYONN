x = 1

print("For testing, x = 1 unless function contains if-statements.\n\
When f(x) contains if-statements, test for possible expected results.")

print("======================================================================")

print("Identity should return x: " + str(identity(x)))
print("Identity derivative should return 1 always: " + str(identity_deriv(0)))
print("Binary step function should return 0 if x < 0: " + str(step(-x)))
print("Binary step function should return 1 if x >= 0: " + str(step(x)))

print("======================================================================")

print("Sigmoid of x using SciPy is: " + str(expit(x)))
print("Sigmoid of x using Numpy is: " + str(logistic(x)))

if expit(x) == logistic(x):
  print("Using SciPy's expit() and Numpy's logistic() give equal results.")
else: 
  print("Using SciPy's expit() and Numpy's logistic() DO NOT give equal results.")

  print("Logistic derivative of x is: " + str(logistic_deriv(x)))

print("======================================================================")

print("Hyperbolic tangent of x is: " + str(tanh(x)))
print("Hyperbolic tangent derivative of x is: " + str(tanh_deriv(x)))
print("ArcTan of x is: " + str(arctan(x)))
print("ArcTan derivative of x is: " + str(arctan(x)))

print("======================================================================")

print("Softsign of x: " + str(softsign(x)))
print("Softsign of -x: " + str(softsign(-x)))
print("Softsign derivative of x: " + str(softsign_deriv(x)))
print("Softsign derivative of -x: " + str(softsign_deriv(-x)))

print("======================================================================")

print("ISRU of x with a = 0.01 is:" +  str(isru(x)))
print("ISRU of x with a = 1.0 is: " +  str(isru(x,1.0)))
print("ISRU derivative of x with a = 0.01 is: " +  str(isru_deriv(x)))
print("ISRU derivative of x with a = 1.0 is: " +  str(isru_deriv(x,1.0)))

print("======================================================================")

print("ISRLU of x with a = 0.01 is: " + str(isrlu(x)))
print("ISRLU of x with a = -1 is: " + str(isrlu(x,-1)))
print("ISRLU of x with a = 0 should be x: " + str(isrlu(x,0)))
print("ISRLU of x with a = 1 should be x: " + str(isrlu(x,1)))
print("ISRLU derivative of x with a = 0.01 should be 1: " + str(isrlu_deriv(x)))
print("ISRLU derivative of x with a = -1 is: " + str(isrlu_deriv(x,-1)))
print("ISRLU derivative of x with a = 1 should be 1: " + str(isrlu_deriv(x,1)))

print("======================================================================")

print("SQNL of x if x = -3.0: " + str(sqnl(-3.0)))
print("SQNL of x if x = -1.0: " + str(sqnl(-1.0)))
print("SQNL of x if x = 1.0: " + str(sqnl(1.0)))
print("SQNL of x if x = 3.0: " + str(sqnl(3.0)))
print("SQNL derivative of x = 1 is: " + str(sqnl_deriv(x)))

print("======================================================================")

print("ReLu of x: " + str(relu(x)))
print("ReLu of x if x = -1.0: " + str(relu(-1.0)))
print("ReLu derivative of x: " + str(relu_deriv(x)))
print("ReLu of x if x = -1.0: " + str(relu_deriv(-1.0)))

print("======================================================================")

print("Leaky ReLu of x: " + str(leaky(x)))
print("Leaky ReLu of x if x = -1.0: " + str(leaky(-1.0)))
print("Leaky ReLu derivative of x: " + str(leaky_deriv(x)))
print("Leaky ReLu derivative of x if x = -1.0: " + str(leaky_deriv(-1.0)))

print("======================================================================")

print("PReLU of x with a = 0.01: " + str(prelu(x)))
print("PReLU of x if x = -1.0 with a = 0.01: " + str(prelu(-1.0)))
print("PReLU derivative of x with a = 0.01: " + str(prelu_deriv(x)))
print("PReLU derivativeof x if x = -1.0 with a = 0.01: " + str(prelu_deriv(-1.0)))

print("======================================================================")

print("RReLU of x with a = 0.01: " + str(rrelu(x)))
print("RReLU of x if x = -1.0 with a = 0.01:" + str(rrelu(-1.0)))
print("RReLU of x if x = -1.0: " + str(rrelu(-1.0)))
print("RReLU derivative of x with a = 0.01: " + str(rrelu_deriv(x)))
print("RReLU derivative of x if x = -1.0 with a = 0.01: " + str(rrelu_deriv(-1.0)))

print("======================================================================")

print("ELU of x : " + str(elu(x)))
print("ELU of x if x = -1.0: " + str(elu(-1.0)))
print("ELU derivative of x: " + str(elu_deriv(x)))
print("ELU derivative of x if x = -1.0: " + str(elu_deriv(-1.0)))

print("======================================================================")

print("SoftPlus of x: " + str(softplus(x)))
print("SoftPlus derivative of x: " + str(softplus_deriv(x)))

print("======================================================================")

print("Bent identity of x: " + str(bentid(x)))
print("Bent identity derivative of x: " + str(bentid_deriv(x)))

print("======================================================================")

print("SoftExponential of x where a = -1.0: " + str(softex(x,-1.0)))
print("SoftExponential of x where a = 0: " + str(softex(x,0)))
print("SoftExponential of x where a = 1.0: " + str(softex(x,1.0)))

print("SoftExponential derivative of x where a = -1.0: " + str(softex_deriv(x,-1.0)))
print("SoftExponential derivative of x where a = 1.0: " + str(softex_deriv(x,1.0)))

print("======================================================================")

print("Soft clipping of x where a = 0.01: " + str(softclip(x)))
print("Soft clipping derivative of x where a = 0.01 and p = 1: " + str(softclip_deriv(x)))

print("======================================================================")

print("Sinusoid of x: " + str(sinusoid(x)))
print("Sinusoid derivative of x: " + str(sinusoid_deriv(x)))

print("======================================================================")

print("Sinc of x where x = 1: " + str(sinc(x)))
print("Sinc of x where x = 0: " + str(sinc(0)))
print("Sinc derivative of x where x = 1: " + str(sinc_deriv(x)))
print("Sinc derivative of x where x = 0: " + str(sinc_deriv(0)))

print("======================================================================")

print("Gaussian of x: " + str(gaussian(x)))
print("Gaussian derivative of x: " + str(gaussian_deriv(x)))
