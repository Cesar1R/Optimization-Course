import numpy as np

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
        x1 = xu - b 
        x3 = xl + b 

        if fun(x1) < fun(x3):
            xu = x3 
            xk = x1 
        else: 
            xl = x1 
            xk = x3 
        
    return xk, fun(xk), xl, xu, k, False 

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


def descenso_max(fun, fun_grad, x0, tol1, tol2, N, Ng, args_sec_dor): 
    '''
    Algoritmo de descenso maximo de gradiente, calcula una punto optimo (minimo) de la funcion fun, con ayuda del gradiente
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