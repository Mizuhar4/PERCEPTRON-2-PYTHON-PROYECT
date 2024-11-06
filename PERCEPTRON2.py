import numpy as np

entrada_dim = 20
oculta_dim = 5
salida_dim = 3

pesos_entrada_oculta = np.random.normal(0, 0.1, (entrada_dim, oculta_dim))
pesos_oculta_salida = np.random.normal(0, 0.1, (oculta_dim, salida_dim))
sesgo_oculta = np.zeros(oculta_dim)
sesgo_salida = np.zeros(salida_dim)

def activacion_sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return x * (1 - x)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

def paso_hacia_adelante(entrada):
    activacion_oculta = activacion_sigmoide(np.dot(entrada, pesos_entrada_oculta) + sesgo_oculta)
    salida = softmax(np.dot(activacion_oculta, pesos_oculta_salida) + sesgo_salida)
    return salida

def generar_trayectoria_lineal():
    trayectoria = np.array([(i, i) for i in range(10)], dtype=float)
    ruido = np.random.normal(0, 0.2, (10, 2))
    trayectoria += ruido
    return trayectoria.flatten()

def generar_trayectoria_circular():
    angulos = np.linspace(0, 2 * np.pi, 10)
    trayectoria = np.array([(5 * np.cos(a), 5 * np.sin(a)) for a in angulos])
    ruido = np.random.normal(0, 0.2, (10, 2))
    trayectoria += ruido
    return trayectoria.flatten()

def generar_trayectoria_aleatoria():
    trayectoria = np.random.uniform(-5, 5, (10, 2))
    return trayectoria.flatten()

def entrenar(entradas, etiquetas, tasa_aprendizaje=0.1, epocas=100):
    global pesos_entrada_oculta, pesos_oculta_salida, sesgo_oculta, sesgo_salida
    for _ in range(epocas):
        for entrada, etiqueta in zip(entradas, etiquetas):
            activacion_oculta = activacion_sigmoide(np.dot(entrada, pesos_entrada_oculta) + sesgo_oculta)
            salida = softmax(np.dot(activacion_oculta, pesos_oculta_salida) + sesgo_salida)

            etiqueta_one_hot = np.zeros(salida_dim)
            etiqueta_one_hot[etiqueta] = 1

            error_salida = salida - etiqueta_one_hot
            error_oculta = np.dot(error_salida, pesos_oculta_salida.T) * derivada_sigmoide(activacion_oculta)

            pesos_oculta_salida -= tasa_aprendizaje * np.outer(activacion_oculta, error_salida)
            pesos_entrada_oculta -= tasa_aprendizaje * np.outer(entrada, error_oculta)
            sesgo_salida -= tasa_aprendizaje * error_salida
            sesgo_oculta -= tasa_aprendizaje * error_oculta

def probar_y_adivinar(num_pruebas=30):
    aciertos = 0
    for _ in range(num_pruebas):
        categoria = np.random.choice([0, 1, 2])
        if categoria == 0:
            trayectoria = generar_trayectoria_lineal()
            etiqueta_real = 0
        elif categoria == 1:
            trayectoria = generar_trayectoria_circular()
            etiqueta_real = 1
        else:
            trayectoria = generar_trayectoria_aleatoria()
            etiqueta_real = 2

        salida = paso_hacia_adelante(trayectoria)
        prediccion = np.argmax(salida)

        print("\nTrayectoria generada:")
        print(trayectoria.reshape(10, 2))

        print(f"\nPredicción del perceptrón: {'Lineal' if prediccion == 0 else 'Circular' if prediccion == 1 else 'Aleatorio'}")
        print(f"Etiqueta real: {'Lineal' if etiqueta_real == 0 else 'Circular' if etiqueta_real == 1 else 'Aleatorio'}")

        if prediccion == etiqueta_real:
            aciertos += 1

    porcentaje_acertacion = (aciertos / num_pruebas) * 100
    print(f"\nPorcentaje de aciertos del perceptrón: {porcentaje_acertacion:.2f}%")

num_ejemplos = 30
datos_linea = [generar_trayectoria_lineal() for _ in range(num_ejemplos)]
datos_circulo = [generar_trayectoria_circular() for _ in range(num_ejemplos)]
datos_aleatorio = [generar_trayectoria_aleatoria() for _ in range(num_ejemplos)]
datos_entrenamiento = datos_linea + datos_circulo + datos_aleatorio
etiquetas_entrenamiento = [0] * num_ejemplos + [1] * num_ejemplos + [2] * num_ejemplos

entrenar(datos_entrenamiento, etiquetas_entrenamiento)
probar_y_adivinar(num_pruebas=30)
