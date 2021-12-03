import numpy,sympy
from sympy import symbols, Eq, solve, log, Matrix, sympify 
import numpy as np

def prep_dimensions(dimensions):
    x=np.array(["x_"+str(i) for i in range(dimensions)])
    input_for_first_line= " ".join(x)
    symb_output = symbols(input_for_first_line) 
    symbols_list = [symb_output[i] for i in range(len(symb_output))]
    
    return symbols_list
    ### this function needs to be commented and expalined 
 


def partial(element, function):
	"""
	partial : sympy.core.symbol.Symbol * sympy.core.add.Add -> sympy.core.add.Add
	partial(element, function) 
    Performs partial derivative of a function of several variables is its derivative with respect to 
    one of those variables, with the others held constant. Return partial_diff.
	"""
	partial_diff = function.diff(element)
    
    # warning is we are not dealing with a function that has an imagineray part 
	#if partial_diff.is_real!=True:     something is wrong here ..... not sure if we need it 
		#print("!careful output was not real")
		#print("partial_diff.is_real",partial_diff.is_real)
		#print("element",element,"function",function)
	return partial_diff


def gradient(partials):
	"""
	gradient : List[sympy.core.add.Add] -> numpy.matrix
	gradient(partials) Transforms a list of sympy objects into a numpy matrix. Return grad.
	"""
	grad = numpy.matrix([partials[i] for i in range(len(partials))])
	return grad

def gradient_to_zero(symbols_list, partials):
	"""
	gradient_to_zero : List[sympy.core.symbol.Symbol] * List[sympy.core.add.Add] -> Dict[sympy.core.numbers.Float]
	gradient_to_zero(symbols_list, partials) Solve the null equation for each variable, and determine the pair of coordinates of the singular point. Return singular.
	"""
	partial_x = Eq(partials[0], 0)
	partial_y = Eq(partials[1], 0)

	singular = solve((partial_x, partial_y), (symbols_list[0], symbols_list[1]))

	return singular

def hessian(symbols_list, function):# modified Gael. Look at input!
	"""
	hessian : List[sympy.core.add.Add] * sympy.core.add.Add -> numpy.matrix
	hessian(partials_second, cross_derivatives) Transforms a list of sympy objects into a numpy hessian matrix. Return hessianmat.
	"""
	hessianmat = numpy.matrix([[partial(symbol_i,partial(symbol_j, function)) for symbol_i in symbols_list] for symbol_j in symbols_list ])

	return hessianmat

def conditiones_fullfilled_and_if_yes_hessian(dimensions,user_input_function_string):
	"""
	Fonction principale.
	"""
    # transform input 
	symbols_list =prep_dimensions(dimensions)
	function = sympify(user_input_function_string)
    
	partials, partials_second = [], []

	for element in symbols_list:
		partial_diff = partial(element, function)
		partials.append(partial_diff)    

	grad = gradient(partials)

	for i in range(0, len(symbols_list)):
		partial_diff = partial(symbols_list[i], partials[i])
		  
 

	hessianmat = Matrix(hessian(symbols_list, function)) 
	isHessianposdef = hessianmat.is_positive_semidefinite
	return isHessianposdef,hessianmat


def hessian_out(dimensions,user_input_function_string):
    try:
        hessian_calculatable= conditiones_fullfilled_and_if_yes_hessian(dimensions,user_input_function_string)
        if hessian_calculatable[0]:
            return (np.array(conditiones_fullfilled_and_if_yes_hessian(dimensions,user_input_function_string)[1])).astype('float')
        else: 
            print("Returned False on whether hessian semi-positive definite")
            return False
    except:
        print("Error found while trying to calculate hessian. Returning that we will not use any hessian.")
        return False


  