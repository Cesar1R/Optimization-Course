import numpy as np
import matplotlib.pyplot as plt

eps = np.finfo('float').eps


#Algoritmo del metodo de la seccion dorada
def met_seccion_dorada(fun, xl, xu, tol, N):
    '''
    Utilizada para calcular el minimo de una funcion en un intervalo 

    Entradas: 
        fun: funcion objetivo 
        xl: extremo izquierdo del intervalo 
        xu: extremo derecho del intervalo 
        tol: Tolerancia utilizada para la condicion de paro 
        N: Numero maximo de interaciones 

    Salidas: 
        xk: Ultimo candidato a punto minimo calculado 
        fun(xk): evaluacion del ultimo candidato en la funcion objetivo
        xl: extremo izquierdo del intervalo final 
        xu: extremo derecho del intervalo final
        k: cantidad de iteraciones completadas por el algoritmo 
        bool: Valor que indica si la condicion de paro se cumplio antes del numero limite de iteraciones (N)
    '''
    rho = (np.sqrt(5) - 1)/2 

    for k in range(1, N+1): 
        if xu - xl < tol: 
            if fun(xl) < fun(xu): 
                xk = xl 
            else: 
                xk = xu 

            return  xk, fun(xk), xl, xu, k, True 
        
        b = rho * (xu - xl) 
        x[0] = xu - b 
        x3 = xl + b 

        if fun(x[0]) < fun(x3):
            xu = x3 
            xk = x[0] 
        else: 
            xl = x[0] 
            xk = x3 
        
    return xk, fun(xk), xl, xu, k, False 


###--------------------Variantes de metodos de descenso maximo de gradiente------------------###

def descenso_max_grad_funciones_cuadradas (A, b, x0, tau, N):
    '''
    Algoritmo que busca estimar un punto optimo para una funcion vectorial cuadratica, de la forma
    f(x) = x.T A x - b.T x
    Entradas: 
        A: Matriz del sistema de ecuaciones Ax = b 
        b: Variable independiente del sistema Ax = b 
        x0: Punto inicial 
        tau: Tolerancia que se utiliza para la condicion de paro 
        N: Numero de iteraciones maximas

    Salidas: 
        xk: Ultimo candidado a punto optimo calculado 
        i: Numero de iteraciones completadas 
        bool: Variable que indica si la condicion de paro se cumplio antes de que se completaran todas las iteraciones, hasta N. 
    '''

    if tau < 0: 
        raise Exception('La tolerancia debe ser no negativa') 
    A = np.asarray(A) 
    b = np.asarray(b) 
    xk = np.asarray(x0)

    for i in range(N+1): 
        #Calculamos el gradiente
        gk = (A @ xk) - b  
        gk = np.array(gk)

        #Comprobamos la condicion de paro 
        if np.linalg.norm(gk, ord=2) < tau: 
            return xk, i, True

        #Obtenemos los valores de alpha_k y x_k 
        alpha_k = (gk.T @ gk) / (gk.T @ A @ gk) 
        
        xk = xk - alpha_k * gk
    return xk, i, False


def descenso_max_GS(fun, fun_grad, x0, tol1, tol2, N, Ng, args_sec_dor): 
    '''
    Algoritmo de descenso maximo de gradiente, calcula una punto optimo (minimo) de la funcion fun, con ayuda del gradiente
    
    Implementacion con -> Seccion dorada

    Entradas 
        fun: Funcion objetivo 
        fun_grad: Gradiente de la funcion objetivo como funcion 
        x0: punto inicial 
        tol1: toleracia 1 
        tol2: tolerancia 2 
        N: Iteraciones maximas para el algoritmo de descenso 
        Ng: Maximo de iteraciones para el metodo de la seccion dorada
        args_sec_dor: Tupla que tiene los argumentos de la seccion dorada

    Salidas: 
        xk: Candidato a minimizador 
        i: Iteraciones completadas 
        bool: Valor que representa si la condicion de paro se cumplio antes de la cantidad maxima de operaciones N 
        pts_sec: Sucesion de puntos que es requerida si dim(x0) = 2. 
    '''
    xk = np.asarray(x0)
    
    if len(x0) == 2: 
        flag = True
        pts_seq = [xk]
    else: 
        flag = False 
        pts_seq = []

    for i in range(N): 
        gk = fun_grad(xk)  

        if np.linalg.norm(gk, ord=2) < tol1: 
            return xk, i, True, pts_seq 

        res_seccion_dorada = met_seccion_dorada(lambda alpha: fun(xk - alpha * fun_grad(xk)), args_sec_dor[0], args_sec_dor[1], tol2, Ng) 
        
        alpha_k = res_seccion_dorada[0] 

        pk = -gk 

        xk = np.asarray(xk + alpha_k * pk)
        
        if flag:
            pts_seq.append(xk.flatten())

    return xk, i, False, pts_seq 



def desc_max_btSUF (f, grad_f, x0, tol = eps, N = 5000, a_init = 1.0, rho = 0.75, c1 = 0.1, Ngs = 500):
    '''
    Algoritmo de descenso maximo de gradiente, calcula una punto optimo (minimo) de la funcion fun, con ayuda del gradiente
    
    Implementacion con -> back trackiing suficiente

    Entradas 
        fun: Funcion objetivo 
        fun_grad: Gradiente de la funcion objetivo como funcion 
        x0: punto inicial 
        tol: toleracia 
        N: Iteraciones maximas para el algoritmo de descenso 
         
    '''
    
    if rho <= 0 or 1 <= rho: 
        raise ValueError("El valor de rho no cumple con las especificaciones")
    
    if c1 <= 0 or 1 <= c1: 
        raise ValueError("El valor de c1 esta fuera del rango permitido") 
    
    if tol <= 0:
        raise ValueError("El valor de la tolerancia es invalido") 

    xk = np.asarray(x0)

    arr = [(x0, a_init, 0)]

    for i in range(N):
        gk = grad_f(xk)
        
        if np.linalg.norm(gk) < tol: 
            return xk, i, True, arr
        
        pk = -gk 

        a, k = back_suficiente(a_init=a_init, rho=rho, c1=c1, xk=xk, f=f, fk=f(xk), grad_xk=gk, pk=pk)

        xk = xk + a*pk

        if len(x0) == 2: 
            arr.append((xk, a, k))

    return xk, N, False, arr 

#Implementacion que utiliza el backtracking con condiciones de wolf
def gradConjugado_noLineal_btWolf (x0, fun, grad_f, N=50000, tau = eps, a_ini=1, rho=0.75, c1=0.001, c2=0.9, Nb=100):
    if tau <= 0:
        raise Exception("Valor incorrecto para tau")
    
    xk = np.asarray(x0)

    delta = eps**(2/3)
    n = len(x0)
    nr = 0 


    gk = grad_f(xk)
    dk = -gk 
    for i in range(N):
        if np.linalg.norm(gk) < tau:
            return xk, gk, i, True, nr
        
        alpha_k = back_tracking_wolf(fun, grad_f, xk, dk, a_ini)
        xk = xk + alpha_k * dk 
        gk_1 = grad_f(xk)
        yk = gk_1 - gk

        if i>0 and i%n > 0 and np.linalg.norm(dk.T @ yk) > delta:
            beta_k = (gk_1.T @ yk)/(dk.T @ yk) 
        else:
            beta_k = 0
            nr += 1

        dk = -gk_1 + beta_k * dk

        if dk.T @ gk_1 >= 0: 
            dk = - gk_1

        gk = gk_1.copy()


    return xk, gk, N, False, nr

def desc_max_btWolf (f, grad_f, x0, tol = eps, N = 5000, a_init = 1.0, rho = 0.75, c1 = 0.1, Ngs = 500):
    '''
    Algoritmo de descenso maximo de gradiente, calcula una punto optimo (minimo) de la funcion fun, con ayuda del gradiente
    
    Implementacion con -> back trackiing suficiente

    Entradas 
        fun: Funcion objetivo 
        fun_grad: Gradiente de la funcion objetivo como funcion 
        x0: punto inicial 
        tol: toleracia 
        N: Iteraciones maximas para el algoritmo de descenso 
         
    '''
    
    if rho <= 0 or 1 <= rho: 
        raise ValueError("El valor de rho no cumple con las especificaciones")
    
    if c1 <= 0 or 1 <= c1: 
        raise ValueError("El valor de c1 esta fuera del rango permitido") 
    
    if tol <= 0:
        raise ValueError("El valor de la tolerancia es invalido") 

    xk = np.asarray(x0)

    arr = [(x0, a_init, 0)]

    for i in range(N):
        gk = grad_f(xk)
        
        if np.linalg.norm(gk) < tol: 
            return xk, i, True, arr
        
        pk = -gk 

        a = back_tracking_wolf(f, grad_f, xk, pk, a_init)

        xk = xk + a*pk

        if len(x0) == 2: 
            arr.append((xk, a, i))

    return xk, N, False, arr 



###--------------------Variantes de backtracking------------------###
def back_tracking_wolf (f, grad_f, x, d, alphaMAX, rho=0.75, c1=0.001, c2=0.9, Nb = 100):
    
    if c1 < 0 or c2 < c1 or 1 < c2:
        raise Exception('Valores erroneos para c1 o c2')

    if rho < 0 or 1 < rho:
        raise Exception("Valor erroneo para rho")

    x = np.asarray(x)
    d = np.asarray(d)
    alpha = alphaMAX

    dot_fd = np.dot(grad_f(x), d)

    for i in range(Nb):
        if f(x + alpha * d) <= f(x) + c1 * alpha * dot_fd and np.dot(grad_f(x + alpha *d), d) >= c2 * dot_fd:
            return alpha
        else: 
            alpha *= rho
    
    return alpha

def back_suficiente(xk, f, fk, grad_xk, pk, a_init = 1.0, rho = 0.75, c1 = 0.1, N=500):
    '''
    Algoritmo de backtracing suficiente
    '''

    if rho <= 0 or 1 <= rho: 
        raise ValueError("El valor de rho no cumple con las especificaciones")
    
    if c1 <= 0 or 1 <= c1: 
        raise ValueError("El valor de c1 esta fuera del rango permitido") 

    if fk == None: 
        fk = f(xk) 
    
    a = a_init
    grad_xk = np.asarray(grad_xk)
    pk = np.asarray(pk) 

    for i in range(N): 
        if f(xk + a * pk) <= f(xk) + c1 * a * grad_xk.T @ pk: 
            return a, i 

        a  = rho*a 

    return float(a), i  



'''
Algunas funciones comunes para hacer pruebas y sus gradientes
'''
def fun_Himmenlblau(x): 
    if len(x) != 2: 
        raise ValueError(f"El vector {x} no es de la dimension correcta")

    return 1.*(x[0]**2 + x[1] - 11)**2 + 1.*(x[0] + x[1] **2 -7)**2 

def grad_Himmenlblau(x): 

    par1 = 2.*(x[0]**2 + x[1] - 11) * (2.* x[0])  + 2.*(x[0] + x[1]**2 - 7) 
    par2 = 2.*(x[0]**2 + x[1] - 11) + 2.* (x[0] + x[1]**2 - 7) * (2. * x[1])
    return np.array([par1, par2])

def hess_fun_Himmenlblau(x):
    if len(x) != 2: 
        raise ValueError(f"El vector {x} no es de la dimension correcta")

    hes_11 = 2 * (2 * (2 * (x[0] **2) + x[0]**2 + x[1] - 11 ) + 1)
    hes_22 = 12 * (x[1] ** 2) + 4 * x[0] - 26 
    hes_12 = 4 * x[0] + 4 * x[1]
    return np.array([[hes_11, hes_12], 
                     [hes_12, hes_22]])

def fun_Beale(x)->float:
    if len(x) != 2: 
        raise ValueError(f"El vector {x} no es de la dimension correcta")
    return (1.5 - x[0] + x[0] * x[1] )**2 + (2.25 - x[0] + x[0] * (x[1] **2)) **2 + (2.625 - x[0] + x[0] * (x[1] **3)) **2

def grad_fun_Beale(x)-> np.ndarray: 
    if len(x) != 2: 
        raise ValueError(f"El vector {x} no es de la dimension correcta")
 
    par_p1_x0 = (1.5 - x[0] + x[0] * x[1] ) * 2. * (-1. + x[1]) 
    par_p2_x0 = (2.25 - x[0] + x[0] * (x[1] **2)) * 2. *  (-1. + x[1] **2) 
    par_p3_x0 = (2.625 - x[0] + x[0] * (x[1] **3)) * 2. * (-1. + x[1] **3) 



    par_p1_x1 = (1.5 - x[0] + x[0] * x[1] ) * 2. * (x[0]) 
    par_p2_x1 = (2.25 - x[0] + x[0] * (x[1] **2)) * 2. *  (x[0] * x[1] * 2)   
    par_p3_x1 = (2.625 - x[0] + x[0] * (x[1] **3)) * 2. * (x[0] * x[1]**2 * 3)



    return np.asarray([(1.5 - x[0] + x[0] * x[1] ) * 2. * (-1. + x[1])  + (2.25 - x[0] + x[0] * (x[1] **2)) * 2. *  (-1. + x[1] **2)  + (2.625 - x[0] + x[0] * (x[1] **3)) * 2. * (-1. + x[1] **3), 
                        (1.5 - x[0] + x[0] * x[1] ) * 2. * (x[0])  + (2.25 - x[0] + x[0] * (x[1] **2)) * 2. *  (x[0] * x[1] * 2)  + (2.625 - x[0] + x[0] * (x[1] **3)) * 2. * (x[0] * x[1]**2 * 3)])

def hess_fun_Beale(x)->np.ndarray:
    if len(x) != 2: 
        raise ValueError(f"El vector {x} no es de la dimension correcta")
    
    hes_f1_p11 = 2.*(-1+x[1])**2
    hes_f2_p11 = 2.*(-1+x[1]**2)**2
    hes_f3_p11 = 2.*(-1+x[1]**3)**2

    hes_f_p11 = hes_f1_p11 + hes_f2_p11 + hes_f3_p11

    hes_f1_p22 = 2.*(x[0]**2)
    hes_f2_p22 = 2.*((2.*x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)*(2.*x[0]))
    hes_f3_p22 = 2.*((3.*x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)*(6.*x[0]*x[1]))

    hes_f_p22 = hes_f1_p22 + hes_f2_p22 + hes_f3_p22

    hes_f1_p12 = 2.*((-1. + x[1])*x[0] + (1.5 - x[0] + x[0]*x[1]))
    hes_f2_p12 = 2.*((-1. + x[1]**2)*(2*x[0]*x[1]) + 2.*(2.25 - x[0] + x[0]*(x[1]**2))*x[1])
    hes_f3_p12 = 2.*((-1. + (x[1]**3))*(3.*x[0]*(x[1]**2)) + (2.625 - x[0] + x[0]*(x[1]**3))*3.*(x[1]**2))

    hes_f_p12 = hes_f1_p12 + hes_f2_p12 + hes_f3_p12

    return np.asarray([[2.*(-1+x[1])**2 + 2.*(-1+x[1]**2)**2 +  2.*(-1+x[1]**3)**2, 2.*((-1. + x[1])*x[0] + (1.5 - x[0] + x[0]*x[1])) + 2.*((-1. + x[1]**2)*(2*x[0]*x[1]) + 2.*(2.25 - x[0] + x[0]*(x[1]**2))*x[1]) + 2.*((-1. + (x[1]**3))*(3.*x[0]*(x[1]**2)) + (2.625 - x[0] + x[0]*(x[1]**3))*3.*(x[1]**2))],
                       [2.*((-1. + x[1])*x[0] + (1.5 - x[0] + x[0]*x[1])) + 2.*((-1. + x[1]**2)*(2*x[0]*x[1]) + 2.*(2.25 - x[0] + x[0]*(x[1]**2))*x[1]) + 2.*((-1. + (x[1]**3))*(3.*x[0]*(x[1]**2)) + (2.625 - x[0] + x[0]*(x[1]**3))*3.*(x[1]**2)), 2.*(x[0]**2) + 2.*((2.*x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)*(2.*x[0])) + 2.*((3.*x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)*(6.*x[0]*x[1]))]])


def fun_Rosenbrock (x): 
    n = len(x) 
    t_sum = 0.0 
    for i in range(n-1): 
        t_sum += 100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 
    return t_sum 

def grad_fun_Rosenbrock(x): 
    n = len(x) 
    grad = np.zeros(len(x))

    grad[0] = -400. * x[0] * (x[1] - x[0]**2) + 2. * (x[0] - 1.)

    for i in range(1, n - 1): 
        grad[i] = -400. * x[i] * (x[i + 1] - x[i]**2) + 2. * (x[i] - 1.) + 200. * (x[i] - x[i - 1]**2)

    grad[-1] = 200. * (x[-1] - x[-2]**2)
    
    return np.asarray(grad)


def hes_fun_Rosenbrock(x):
    n = len(x)
    hes = np.zeros((n, n))

    hes[0, 0] = 1200. * (x[0]**2) - 400. * x[1] + 2 
    hes[0, 1] = -400.*x[0]

    for i in range(1, n-1):
            hes[i, i] = 1200 * (x[i]**2) - 400 * x[i+1] + 202
            hes[i, i-1] = -400 * x[i-1]
            hes[i, i+1] = -400 * x[i]

    hes[n-1, n-2] = -400 * x[n-2] 
    hes[n-1, n-1] = 200

    return hes


alpha_Hart = np.array([1.0, 1.2, 3.0, 3.2])
a_Hart = np.array([
    [10, 3, 17, 3, 3.5, 1],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1, 10, 10, 1],
    [17, 8, 0.05, 10, 0.1, 14]
])
p_Hart = np.array([
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.3943, 0.7582, 0.1445, 0.9124, 0.6618, 0.3797],
    [0.5951, 0.1996, 0.7220, 0.8641, 0.4123, 0.8828]
])

def fun_Hartman(x):
    sum_ = 0.0
    for i in range(4):
        sum_ += alpha_Hart[i] * np.exp(-np.sum(a_Hart[i, :] * (x - p_Hart[i, :])**2))
    
    return - (1./1.94) * (2.58 + sum_)

def grad_fun_Hartman(x):
    n = len(x)
    grad = np.zeros(n)
    for k in range(n):
        tmp_grad = 0
        for i in range(4):
            suma_ = np.exp(-np.sum(a_Hart[i, :] * (x-p_Hart[i, :])**2))
            tmp_grad += 2. * alpha_Hart[i] * a_Hart[i, k] * (x[k] - p_Hart[i, k]) * suma_
        grad[k] = tmp_grad/1.94
    return grad

def hes_fun_Hartman(x):
    n = len(x)
    hess = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            tmp_hes = 0.0
            for i in range(4):
                sum_ = np.exp(-np.sum(a_Hart[i, :] * (x - p_Hart[i, :])**2))
                tmp_hes += alpha_Hart[i] * sum_ * (2 * a_Hart[i, k] * (k == j) - 4 * a_Hart[i, k] * a_Hart[i, j] * (x[k] - p_Hart[i, k]) * (x[j] - p_Hart[i, j]))
                
            hess[k, j] = tmp_hes/1.94
    return hess


'''
Funcion para minumos cuadrados y sistemas de ecuaciones
'''
def solve_sistema (X, y, n): ##Tarea 8

    matriz = np.dot(X.T, X)
    vector = np.dot(X.T, y) 

    return np.linalg.solve(matriz, vector), np.linalg.cond(X)

#Raiz del error cuadratico medio entre los valores verdaderos y los que predice el modelo
def RMSE (y, y_pred, m): ## Tarea 8
    return np.sqrt(np.dot(y - y_pred, y - y_pred)/m)


####---------------------------> Minimos Cuadrados <--------------------------------------####
#---Algoritmo de Levenberg Marquart para minimos cuadrados---#
eps = np.finfo('float').eps 
tol_ = np.sqrt(m) * eps**(1/3)

##Tarea 8
def lev_marq (fun_res, fun_jac, z0, N=200, mu=0.001, tol=tol_):
    if mu <= 0: 
        raise ValueError("Valor incorrecto de mu")

    if tol <= 0: 
        raise ValueError("Valor incorrecto para la tolerancia")


    zk =  z0
    res = 0 
    mat_id = np.eye(len(zk))

    Rk = np.array(fun_res(zk))
    Jk = np.array(fun_jac(zk))
    fk = 0.5 * np.dot(Rk.T,  Rk)
    A = np.dot(Jk.T, Jk)
    g = np.dot(Jk.T, Rk)


    mu = min(mu, max(np.diag(A)))


    flag = False 

    for k in range(N):
        pk = np.linalg.solve((A + mu * mat_id), -g)


        if np.linalg.norm(pk) < tol:
            flag = True
            break

        zk1 = zk + pk
        Rk1 = fun_res(zk1)
        fk1 = 0.5 * np.dot(Rk1.T, Rk1)
         
        rho = (fk - fk1)/(np.dot(np.dot(pk.T, Jk.T), Rk) + 0.5 * mu * np.dot(pk.T, pk))

        if rho < 0.25: 
            mu = 2 * mu
        elif rho > 0.75:
            mu = mu/3


        Rk = Rk1 
        fk = fk1 
        zk = zk1

        Jk = fun_jac(zk1)
        A = np.dot(Jk.T, Jk)
        g = np.dot(Jk.T, Rk)

    return zk, fk, k, flag


#Metodo de newton 
##Tarea 6

def metNewton (f, grad_f, hes_f, x0, N=10000, tol=1e-8, a_ini =1, c1=0.1, rho=0.6, Nb=100):
    if tol < 0:
        raise ValueError(f"Valor invalido para la tolerancia {tol}")
    n = len(x0)
    m = 0
    x = np.asarray(x0)

    gk = grad_f(x)
    hk = hes_f(x)
    sec = []
    for k in range(N):

        if np.linalg.norm(gk) < tol:
            return x, gk, k, m, True, sec

        
        try:
            c, low = sc.linalg.cho_factor(hk)
            pk = sc.linalg.cho_solve((c, low), -gk)
        except sc.linalg.LinAlgError:
            pk = -np.asarray(gk)
            m = m+1

        alpha, _= back_suficiente(x, f, f(x), gk, pk, a_init=a_ini, c1=c1, rho=rho, N=Nb)

        x = x + alpha * pk 
        gk = grad_f(x)
        hk = hes_f(x)
        if n == 2:
            sec.append(x)

    return x, gk, N, m, False, sec

## Metodo de newton sin la factorizacion de cholesky
def metNewton_eigens (f, grad_f, hes_f, x0, delta, N=10000, tol=1e-8, a_ini =1, c1=0.1, rho=0.6, Nb=100):
    if tol < 0:
        raise ValueError(f"Valor invalido para la tolerancia {tol}")
    if delta < 0:
        raise ValueError(f"Valor invalido para el incremento {delta}")

    sqrt_eps = np.sqrt(np.finfo(float).eps)

    n = len(x0)
    x = np.asarray(x0)
    for k in range(N):
        gk = grad_f(x)

        if np.linalg.norm(gk) < tol:
            return x, gk, k, True
        
        Hk = hes_f(x)
        l, _ = get_max_min_eigenval(Hk)

        if l < sqrt_eps:
            mod = delta + abs(l)
            Hk = Hk + mod*np.eye(n)
            
        try:
            c, low = sc.linalg.cho_factor(Hk)
            pk = sc.linalg.cho_solve((c, low), -gk)
        except sc.linalg.LinAlgError:
            print("La matriz no es definida positiva")

        alpha, _= back_suficiente(x, f, f(x), gk, pk, a_init=a_ini, c1=c1, rho=rho, N=Nb)
        
        x = x + alpha*pk

    return x, gk, N, False


#Funciones relacionadas con matrices, eigenvalores e igenvectores 
def get_max_min_eigenval(A):
    eigens = np.linalg.eigvalsh(A)
    eigens = np.sort(eigens)
    return eigens[0], eigens[-1]

def its_hermitian(A, lmin, lmax):
    if lmin > 0:
        print("Matriz definida positiva")
    elif lmax < 0:
        print("Matriz definida negativa")
    else:
        print("Matriz indefinida")

'''
Funciones de utilidad para extraer y mostrar los datos 
'''


#Extraccion de datos para la Tarea 3 
#Regresa un arreglo con los puntos iniciales, la media de los a y de los i


def extraccion_Tarea3(arr):

    suc_puntos = []
    arr_a = []
    arr_i = [] 

    for p, a, i in arr: 
        suc_puntos.append(p)
        arr_a.append(a)
        arr_i.append(i) 

    media_a = np.mean(arr_a)
    media_i = np.mean(arr_i)

    return suc_puntos, media_a, media_i



def trayectorias2D (arr):
    x = [pt[0] for pt in arr]
    y = [pt[1] for pt in arr] 

    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title("Trayectorias en 2D") 

    plt.show()


def plot2DFnc(fncf, xleft, xright, ybottom, ytop, levels=None):
    # Crea una discretización uniforme del intervalo [xleft, xright]
    ax = np.linspace(xleft, xright, 250)
    # Crea una discretización uniforme del intervalo [ybottom, ytop]
    ay = np.linspace(ybottom, ytop, 250)
    # La matriz mX que tiene las abscisas 
    mX, mY = np.meshgrid(ax, ay)
    # Se crea el arreglo mZ con los valores de la función en cada nodo
    vz = np.zeros( len(ax)*len(ay) )
    for i,xy in enumerate(zip(mX.flatten(), mY.flatten())):
        vz[i] = fncf(xy)
    mZ = vz.reshape( len(ay), len(ax) ) 
    if levels is None:
        fig = plt.figure(figsize=(6,4))
        im = plt.imshow(mZ, cmap='coolwarm', extent=(xleft, xright, ybottom, ytop), 
                        interpolation='bilinear', origin='lower') 
        plt.colorbar(im)
    else:
        fig, ax = plt.subplots()
        CS = ax.contour(mX, mY, mZ, levels, cmap='RdGy', linestyles='dashed', linewidths=0.75)



def load_data (path): 
    data = np.load(path) 
    A = data['arr_0']
    b = data['arr_1'] 
    return A, b 

def get_condicion(A):
    A = np.asarray(A) 

    eigenvalores = np.linalg.eigvals(A) 

    eigenvalores.sort()

    return eigenvalores[-1]/eigenvalores[0]

'''
Funciones 
'''


    