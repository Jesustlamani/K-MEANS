import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog

# Función para cargar datos desde un archivo CSV
def cargar_csv():
    archivo_csv = filedialog.askopenfilename(title="Seleccionar archivo CSV", filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")))
    if archivo_csv:
        return pd.read_csv(archivo_csv)
    else:
        return None

# Función principal para visualizar los datos
def visualizar_datos():
    df_filled = cargar_csv()
    if df_filled is not None:
        cat_atts = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19]
        num_atts = [1, 14, 15, 16, 17, 18]
        data = df_filled.values
        
        # Obtener el número de centroides seleccionado por el usuario
        num_centroides = var_num_centroides.get()
        
        # Definir funciones de distancia y algoritmo K-Means (tus funciones aquí)
        def heom_distance(instance1, instance2, cat_atts, num_atts):
            """
            Calcula la distancia HEOM entre dos instancias.
            """
            sum_dist = 0

            # Calcular distancia para atributos categóricos
            for cat_att in cat_atts:
                if instance1[cat_att] != instance2[cat_att]:
                    sum_dist += 1

            # Calcular distancia para atributos numéricos
            for num_att in num_atts:
                sum_dist += (instance1[num_att] - instance2[num_att]) ** 2

            heom_dist = np.sqrt(sum_dist)
            return heom_dist

        def heom_distances(data, cat_atts, num_atts):
            num_instances = len(data)
            distances = np.zeros((num_instances, num_instances))

            for i in range(num_instances):
                for j in range(num_instances):
                    distances[i][j] = heom_distance(data[i], data[j], cat_atts, num_atts)

            return distances


        def k_means(data, k, max_iterations=5):
            num_instances, num_features = data.shape

            # Inicializar centroides aleatorios
            centroids = data[np.random.choice(num_instances, size=k, replace=False)]

            # Historial de centroides para visualización
            centroids_history = [centroids.copy()]

            for _ in range(max_iterations):
                # Asignar instancias a clusters más cercanos
                clusters = [[] for _ in range(k)]
                for instance in data:
                    distances = [heom_distance(instance, centroid, cat_atts, num_atts) for centroid in centroids]
                    cluster_index = np.argmin(distances)
                    clusters[cluster_index].append(instance)

                # Actualizar centroides
                new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]

                # Agregar nuevos centroides al historial
                centroids_history.append(new_centroids.copy())

                # Verificar convergencia
                if np.allclose(centroids, new_centroids):
                    break

                centroids = new_centroids

            return clusters, centroids, centroids_history
        
        heom_distances = heom_distances(data, cat_atts, num_atts)
        
        # Definir el número de centroides
        k = num_centroides
        
        # Ejecutar el algoritmo K-Means
        clusters, centroids, centroids_history = k_means(heom_distances, k)
        
        # Asignar colores a los centroides y a los puntos asociados
        colores_centroides = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # Lista de colores para los centroides

        # Visualización de datos
        plt.figure(figsize=(12, 9))
        for i in range(1, len(centroids_history)):
            plt.subplot(2, 3, i)
            for idx, cluster in enumerate(clusters):
                cluster_np = np.array(cluster)
                color = colores_centroides[idx]  # Asignamos un color único a cada centroide
                plt.scatter(cluster_np[:, 0], cluster_np[:, 1], color=color, alpha=0.5)
            for j in range(len(centroids_history[i])):
                centroid_x, centroid_y = centroids_history[i][j][0], centroids_history[i][j][1]
                plt.plot(
                    centroid_x,
                    centroid_y,
                    marker="o",
                    markersize=15,  # Tamaño de los marcadores
                    label=f"Centroide {j+1}",
                    color=colores_centroides[j % 10],  # Asignamos un color único a cada centroide
                )
            plt.xlabel("Feature 1", fontsize=20)  # Tamaño de la fuente
            plt.ylabel("Feature 2", fontsize=20)  # Tamaño de la fuente
            plt.title(f"Iteración {i}", fontsize=24)  # Tamaño de la fuente
            plt.grid(True)
            plt.tight_layout()

        plt.show()

# Función para salir de la aplicación
def salir():
    root.quit()

# Crear la ventana principal
root = Tk()
root.title("Visualización de Datos")
root.geometry("800x600")  # Tamaño de la ventana
root.configure(bg='#008000')  # Color de fondo verde

# Etiqueta para el texto "ALGORITMO KMEANS"
etiqueta = Label(root, text="ALGORITMO KMEANS", font=("Arial", 60, "bold"), bg='#008000', fg='black')  # Tamaño de la fuente, negrita y color de fondo
etiqueta.pack(pady=30)  # Espacio vertical

# Widget de selección para el número de centroides
var_num_centroides = IntVar()
var_num_centroides.set(3)  # Valor predeterminado

lbl_num_centroides = Label(root, text="Número de centroides:", font=("Arial", 20), bg='#008000', fg='black')  # Tamaño de la fuente y color de fondo
lbl_num_centroides.pack()

rbtn_centroides_3 = Radiobutton(root, text="3", variable=var_num_centroides, value=3, font=("Arial", 18), bg='#008000', fg='black')  # Tamaño de la fuente y color de fondo
rbtn_centroides_3.pack()

rbtn_centroides_5 = Radiobutton(root, text="5", variable=var_num_centroides, value=5, font=("Arial", 18), bg='#008000', fg='black')  # Tamaño de la fuente y color de fondo
rbtn_centroides_5.pack()

rbtn_centroides_10 = Radiobutton(root, text="10", variable=var_num_centroides, value=10, font=("Arial", 18), bg='#008000', fg='black')  # Tamaño de la fuente y color de fondo
rbtn_centroides_10.pack()

# Crear un botón para cargar el archivo CSV
btn_cargar_csv = Button(root, text="Cargar archivo CSV", command=visualizar_datos, bg="#007bff", fg="white", font=("Arial", 24))  # Tamaño de la fuente y color de fondo
btn_cargar_csv.pack(pady=20)  # Espacio vertical

# Crear un botón para salir de la aplicación
btn_salir = Button(root, text="Salir", command=salir, font=("Arial", 24), bg="#dc3545", fg="white")  # Tamaño de la fuente y color de fondo
btn_salir.pack(pady=20)  # Espacio vertical

# Ejecutar el bucle principal de la ventana
root.mainloop()
