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
